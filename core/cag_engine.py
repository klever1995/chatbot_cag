# core/cag_engine.py
import logging
from models.llms import azure_llm_client
from cache.context_cache import context_cache
from cache.redis_manager import redis_cache

logger = logging.getLogger(__name__)

class CAGEngine:
    """Motor CAG para respuestas eficientes desde caché de contexto"""
    
    def __init__(self):
        logger.info("✅ Motor CAG inicializado")
    
    def generate_response(self, query):
        """Genera una respuesta usando el contexto cachead"""
        try:
            # 1. Buscar en caché de contexto
            logger.debug(f"🔍 Buscando en caché de contexto: {query}")
            context_results = context_cache.search_in_cache(query)
            
            if not context_results:
                return "No tengo información suficiente en mi conocimiento almacenado para responder esta pregunta."
            
            # 2. Construir contexto con los documentos encontrados
            context = self._build_context(context_results)
            
            # 3. Generar prompt para el LLM
            messages = self._build_messages(query, context)
            
            # 4. Obtener respuesta del LLM
            logger.debug("🧠 Generando respuesta con Azure OpenAI...")
            response = azure_llm_client.generate_response(messages)
            
            # 5. Cachear respuesta en Redis
            redis_cache.set_cached_response(query, response)
            
            logger.debug("✅ Respuesta CAG generada exitosamente")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error en el motor CAG: {str(e)}")
            return "Lo siento, ocurrió un error al procesar tu consulta desde la caché."
    
    def _build_context(self, context_results):
        """Construye el contexto a partir de los resultados de caché"""
        context = ""
        
        for i, result in enumerate(context_results, 1):
            context += f"[Documento {i}]: {result['document']}\n\n"
        
        return context
    
    def _build_messages(self, query, context):
        """Construye los mensajes para el LLM con el contexto cachead"""
        system_message = """Eres un asistente útil que responde preguntas basándose ÚNICAMENTE en el contexto proporcionado desde tu conocimiento almacenado. 
Sigue estas reglas:
1. Responde solo con la información del contexto
2. Si la información no está en el contexto, di que no lo sabes
3. Sé conciso y directo
4. Usa el mismo idioma de la pregunta"""

        user_message = f"""Contexto de conocimiento almacenado:
{context}

Pregunta: {query}

Por favor, responde basándote solo en el contexto anterior de tu conocimiento almacenado."""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def process_query(self, query):
        """Método principal para procesar consultas CAG"""
        return self.generate_response(query)

# Instancia singleton para ser importada
cag_engine = CAGEngine()