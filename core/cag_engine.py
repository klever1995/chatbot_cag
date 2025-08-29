# core/cag_engine.py
import logging
import re
from typing import List, Dict, Any
from models.llms import azure_llm_client
from cache.redis_manager import redis_cache

logger = logging.getLogger(__name__)

class CAGEngine:
    """Motor CAG optimizado para respuestas con contexto completo"""

    def __init__(self):
        logger.info("Motor CAG optimizado inicializado")

    def generate_response(self, query: str, external_context: str = None) -> str:
        """
        Genera una respuesta usando contexto externo desde RAG.
        
        Parámetros:
            query (str): Pregunta del usuario.
            external_context (str, opcional): Contexto completo pasado desde RAG.
        """
        try:
            # 1. Verificar si existe respuesta en Redis
            cached_response = redis_cache.get_cached_response(query)
            if cached_response:
                logger.debug(f"Respuesta recuperada de Redis: {query}")
                return cached_response

            # 2. Validar que tenemos contexto externo
            if not external_context or not external_context.strip():
                logger.debug("No se proporcionó contexto externo válido")
                return "No tengo información suficiente para responder esta pregunta."

            # 3. Optimizar el contexto para no exceder límites de tokens
            optimized_context = self._optimize_context(query, external_context)
            
            if not optimized_context:
                logger.debug("No se pudo optimizar el contexto para la consulta")
                return "No lo sé"

            # 4. Generar prompt para el LLM
            messages = self._build_messages(query, optimized_context)

            # 5. Obtener respuesta del LLM con reintentos
            logger.debug("Generando respuesta con Azure OpenAI...")
            response = azure_llm_client.generate_response(messages)

            # 6. Cachear respuesta en Redis SOLO si es una respuesta válida
            if response and not self._is_negative_response(response):
                redis_cache.set_cached_response(query, response)
                logger.debug("Respuesta válida cacheada en Redis")
            else:
                logger.debug("Respuesta negativa NO cacheada en Redis")

            logger.info("Respuesta CAG generada exitosamente")
            return response

        except Exception as e:
            logger.error(f"Error en el motor CAG: {str(e)}")
            return "Lo siento, ocurrió un error al procesar tu consulta."

    def _optimize_context(self, query: str, context: str) -> str:
        """
        Optimiza el contexto para enfocarse en las partes más relevantes
        usando búsqueda de términos clave antes de enviar al LLM.
        """
        try:
            # Si el contexto es manejable, usarlo completo
            if len(context) <= 8000:
                return context
            
            # Para contextos muy grandes, buscar secciones relevantes
            logger.debug(f"Optimizando contexto grande ({len(context)} caracteres)")
            
            # Buscar términos clave de la consulta en el contexto
            query_terms = self._extract_key_terms(query)
            relevant_sections = []
            
            # Dividir el contexto por documentos o secciones
            sections = re.split(r'\[DOCUMENTO:[^\]]+\]', context)
            
            for section in sections:
                if not section.strip():
                    continue
                    
                # Calcular relevancia de la sección
                relevance_score = self._calculate_section_relevance(section, query_terms)
                
                if relevance_score > 0.3:  # Umbral de relevancia
                    relevant_sections.append((section, relevance_score))
            
            # Ordenar por relevancia y tomar las mejores
            relevant_sections.sort(key=lambda x: x[1], reverse=True)
            
            # Construir contexto optimizado
            optimized_context = ""
            total_length = 0
            max_length = 10000  # Límite seguro para el LLM
            
            for section, score in relevant_sections:
                if total_length + len(section) <= max_length:
                    optimized_context += f"\n\n[Relevancia: {score:.2f}]\n{section}"
                    total_length += len(section)
                else:
                    break
            
            if not optimized_context:
                # Fallback: tomar el inicio del contexto
                optimized_context = context[:8000] + "\n\n[...]"
                logger.debug("Usando fallback de contexto truncado")
            
            logger.debug(f"Contexto optimizado: {len(optimized_context)} caracteres")
            return optimized_context
            
        except Exception as e:
            logger.error(f"Error optimizando contexto: {str(e)}")
            return context[:10000]  # Fallback seguro

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extrae términos clave de la consulta"""
        # Eliminar stopwords y términos comunes
        stopwords = {"qué", "cómo", "cuándo", "dónde", "por", "qué", "para", "puedo", "se", "puede"}
        words = re.findall(r'\b[a-záéíóúñ]+\b', query.lower())
        return [word for word in words if word not in stopwords and len(word) > 2]

    def _calculate_section_relevance(self, section: str, query_terms: List[str]) -> float:
        """Calcula la relevancia de una sección para los términos de la consulta"""
        if not query_terms:
            return 0.0
            
        section_lower = section.lower()
        matches = 0
        
        for term in query_terms:
            if term in section_lower:
                matches += 1
                # Bonus por múltiples ocurrencias
                matches += section_lower.count(term) * 0.1
        
        return matches / len(query_terms)

    def _is_negative_response(self, response: str) -> bool:
        """Determina si la respuesta es negativa (no debe cachearse)"""
        if not response or not response.strip():
            return True
            
        negative_phrases = [
            "no lo sé",
            "no tengo información",
            "no está en el contexto",
            "no encuentro",
            "no se menciona",
            "no aparece",
            "no puedo responder",
            "no hay información",
            "la información no está disponible",
            "no tengo datos"
        ]
        
        response_lower = response.lower().strip()
        
        # Verificar si la respuesta es exactamente "No lo sé"
        if response_lower == "no lo sé":
            return True
            
        # Verificar si contiene frases negativas
        return any(phrase in response_lower for phrase in negative_phrases)

    def _build_messages(self, query: str, context: str) -> List[Dict]:
        """Construye los mensajes para el LLM con el contexto optimizado"""
        system_message = """Eres un asistente especializado en documentación legal y corporativa. 
Responde preguntas basándote ÚNICAMENTE en el contexto proporcionado.

REGLAS ESTRICTAS:
1. Responde SOLO con la información del contexto proporcionado
2. Si la información no está en el contexto, di EXACTAMENTE "No lo sé"
3. Sé preciso, conciso y directo
4. Cita artículos, secciones o documentos cuando sea posible
5. Usa el mismo idioma de la pregunta
6. No inventes información bajo ninguna circunstancia"""

        user_message = f"""CONTEXTO COMPLETO:
{context}

PREGUNTA: {query}

INSTRUCCIÓN: Responde basándote ÚNICAMENTE en el contexto anterior. Si la respuesta no está en el contexto, di EXACTAMENTE "No lo sé"."""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    def process_query(self, query: str, external_context: str = None) -> str:
        """Método principal para procesar consultas CAG"""
        return self.generate_response(query, external_context=external_context)

# Instancia singleton para ser importada
cag_engine = CAGEngine()