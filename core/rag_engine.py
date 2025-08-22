# core/rag_engine.py (versión corregida)
import logging
from models.llms import azure_llm_client
from databases.chroma_client import chroma_client
from cache.redis_manager import redis_cache

logger = logging.getLogger(__name__)

class RAGEngine:
    """Motor RAG para consultas complejas con búsqueda en ChromaDB"""
    
    def __init__(self):
        logger.info("✅ Motor RAG inicializado")
    
    def generate_response(self, query, n_results=5):
        """Genera una respuesta usando el pipeline RAG completo"""
        try:
            # 1. Búsqueda en ChromaDB
            logger.debug(f"🔍 Buscando en ChromaDB: {query}")
            search_results = chroma_client.search(query, n_results=n_results)
            
            # Verificar si hay resultados válidos
            if (not search_results or 
                not search_results.get('documents') or 
                not search_results['documents'][0]):
                logger.warning("⚠️ No se encontraron documentos relevantes")
                return "No encontré información relevante para tu consulta."
            
            # 2. Construir contexto con los documentos encontrados
            context = self._build_context(search_results)
            logger.debug(f"📋 Contexto construido con {len(search_results['documents'][0])} documentos")
            
            # 3. Generar prompt para el LLM
            messages = self._build_messages(query, context)
            
            # 4. Obtener respuesta del LLM
            logger.debug("🧠 Generando respuesta con Azure OpenAI...")
            response = azure_llm_client.generate_response(messages)
            
            # 5. Cachear respuesta en Redis
            redis_cache.set_cached_response(query, response)
            
            logger.info("✅ Respuesta RAG generada exitosamente")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error en el motor RAG: {str(e)}", exc_info=True)
            return "Lo siento, ocurrió un error al procesar tu consulta."
    
    def _build_context(self, search_results):
        """Construye el contexto a partir de los resultados de búsqueda"""
        context = ""
        documents = search_results['documents'][0]
        
        for i, doc in enumerate(documents, 1):
            context += f"[Documento {i}]: {doc}\n\n"
        
        return context
    
    def _build_messages(self, query, context):
        """Construye los mensajes para el LLM con el contexto"""
        system_message = """Eres un asistente especializado que responde preguntas basándose ÚNICAMENTE en el contexto proporcionado. 

REGLAS ESTRICTAS:
1. Responde SOLO con la información del contexto proporcionado
2. Si la información no está en el contexto, di: "No tengo información sobre esto en mis documentos"
3. No inventes información bajo ninguna circunstancia
4. Sé conciso pero informativo
5. Usa oraciones completas y párrafos coherentes
6. Responde en el mismo idioma que la pregunta"""

        user_message = f"""CONTEXTO PROPORCIONADO:
{context}

PREGUNTA: {query}

INSTRUCCIÓN: Responde la pregunta usando EXCLUSIVAMENTE la información del contexto anterior. Si la respuesta no está en el contexto, di específicamente "No tengo información sobre esto en mis documentos"."""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def process_query(self, query):
        """Método principal para procesar consultas RAG"""
        return self.generate_response(query)

# Instancia singleton para ser importada
rag_engine = RAGEngine()