import logging
from models.llms import azure_llm_client
from cache.context_cache import context_cache
from cache.redis_manager import redis_cache

logger = logging.getLogger(__name__)

class CAGEngine:
    """Motor CAG para respuestas eficientes desde caché de contexto y conversación general"""
    
    def __init__(self):
        logger.info("✅ Motor CAG inicializado")
    
    def generate_response(self, query, is_general_conversation=False):
        """Genera una respuesta usando el contexto cacheado o conversación general"""
        try:
            # Conversación general (sin caché de documentos)
            if is_general_conversation:
                logger.debug(f"💬 Modo conversación general: {query}")
                return self._generate_general_response(query)
            
            #Búsqueda en caché de contexto documental
            logger.debug(f"🔍 Buscando en caché de contexto: {query}")
            context_results = context_cache.search_in_cache(query)
            
            if not context_results:
                return "No tengo información suficiente en mi conocimiento almacenado para responder esta pregunta."
            
            # Construir contexto con los documentos encontrados
            context = self._build_context(context_results)
            
            # Generar prompt para el LLM con contexto documental
            messages = self._build_document_context_messages(query, context)
            
            # Obtener respuesta del LLM optimizada para contexto legal
            logger.debug("Generando respuesta legal con Azure OpenAI...")
            response = azure_llm_client.generate_legal_response(messages)
            
            # Cachear respuesta en Redis
            redis_cache.set_cached_response(query, response)
            
            logger.debug("✅ Respuesta CAG generada exitosamente")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error en el motor CAG: {str(e)}")
            return "Lo siento, ocurrió un error al procesar tu consulta."
    
    def _generate_general_response(self, query):
        """Genera respuestas para conversación general"""
        general_messages = [
            {
                "role": "system", 
                "content": """Eres un asistente amable y útil especializado en temas laborales y legales. 
Responde preguntas generales de manera cortés pero mantén el foco en tu área de expertise.
Para preguntas muy personales o fuera de contexto, redirige amablemente a temas laborales.
Mantén respuestas breves y naturales."""
            },
            {
                "role": "user", 
                "content": query
            }
        ]
        
        try:
            # Método optimizado para conversación
            response = azure_llm_client.generate_conversational_response(general_messages)
            redis_cache.set_cached_response(query, response)
            return response
        except Exception as e:
            logger.error(f"❌ Error en respuesta general: {str(e)}")
            return "¡Hola! Soy un asistente especializado en temas laborales y documentación. ¿En qué puedo ayudarte hoy?"
    
    def _build_context(self, context_results):
        """Construye el contexto a partir de los resultados de caché"""
        if not context_results:
            return "No se encontró contexto relevante."
        
        context = "📚 CONTEXTO ENCONTRADO EN KNOWLEDGE BASE:\n\n"
        
        for i, result in enumerate(context_results, 1):
            similarity = result.get('similarity', 0)
            context += f"--- Documento {i} (Relevancia: {similarity:.0%}) ---\n"
            context += f"{result['document']}\n\n"
        
        return context
    
    def _build_document_context_messages(self, query, context):
        """Construye los mensajes para el LLM con el contexto cacheado"""
        system_message = """Eres un asistente legal/laboral especializado. Responde preguntas basándote ÚNICAMENTE en el contexto proporcionado.

REGLAS ESTRICTAS:
1. Responde EXCLUSIVAMENTE con información del contexto proporcionado
2. Si la información no está en el contexto, di: "No tengo información sobre esto en mi knowledge base"
3. Sé preciso, conciso y profesional
4. Usa el mismo idioma de la pregunta
5. Cita la relevancia del documento cuando sea apropiado
6. Mantén un tono profesional pero accesible

IMPORTANTE: Nunca inventes información o hagas suposiciones fuera del contexto."""

        user_message = f"""CONTEXTO DE KNOWLEDGE BASE:
{context}

PREGUNTA: {query}

INSTRUCCIÓN: Responde la pregunta basándote ÚNICAMENTE en el contexto anterior. Si la respuesta no está en el contexto, indícalo claramente."""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def _handle_empty_context(self, query):
        """Maneja el caso cuando no se encuentra contexto relevante"""
        no_context_messages = [
            {
                "role": "system",
                "content": """Eres un asistente especializado en temas laborales. 
Cuando no tienes información en tu knowledge base, responde amablemente indicando esto y ofrece ayuda dentro de tu área de expertise."""
            },
            {
                "role": "user",
                "content": f"Pregunta: {query}\n\nInstrucción: Responde indicando que no tienes información específica sobre esto en tu knowledge base, pero mantén un tono útil."
            }
        ]
        
        try:
            return azure_llm_client.generate_conversational_response(no_context_messages)
        except Exception:
            return "No tengo información específica sobre esto en mi knowledge base. ¿Puedo ayudarte con alguna otra consulta laboral?"

    def process_query(self, query, is_general_conversation=False):
        """Método principal para procesar consultas CAG"""
        return self.generate_response(query, is_general_conversation)

# Instancia singleton para ser importada
cag_engine = CAGEngine()