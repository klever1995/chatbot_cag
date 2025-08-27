# core/orchestrator.py
import logging
from cache.redis_manager import redis_cache
from core.rag_engine import rag_engine
from core.cag_engine import cag_engine

logger = logging.getLogger(__name__)

class Orchestrator:
    """Orquestador inteligente con flujo universal Redis → RAG → CAG"""

    def __init__(self):
        logger.info("✅ Orquestador universal inicializado")

    def _should_fallback_to_cag(self, response):
        """Determina si la respuesta del RAG requiere fallback a CAG"""
        if not response or not response.strip():
            return True
        
        negative_indicators = [
            "no lo sé",
            "no encuentro",
            "no tengo información",
            "no está en el contexto",
            "no aparece en el texto",
            "no se menciona"
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in negative_indicators)

    def _is_negative_response(self, response):
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
            "no puedo responder"
        ]
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in negative_phrases)

    def _cache_response_if_valid(self, query, response):
        """Cachea la respuesta solo si no es negativa"""
        if not self._is_negative_response(response):
            redis_cache.set_cached_response(query, response)
            logger.debug("✅ Respuesta válida cacheada en Redis")
        else:
            logger.debug("⏩ Respuesta negativa NO cacheada en Redis")

    def route_query(self, query):
        """
        Ruta universal: Redis → RAG → CAG para TODAS las consultas
        Returns:
            dict: {"response": str, "route": str, "source": str}
        """
        try:
            # 1️⃣ Revisar si ya existe en Redis
            cached = redis_cache.get_cached_response(query)
            if cached:
                logger.info("✅ Respuesta obtenida desde Redis")
                return {"response": cached, "route": "redis", "source": "redis_cache"}

            # 2️⃣ Siempre usar RAG primero
            logger.debug("🔍 Ruta universal → RAG primero")
            response = rag_engine.generate_answer(query)
                
            if not self._should_fallback_to_cag(response):
                # RAG tuvo éxito → Cachear si es válida
                self._cache_response_if_valid(query, response)
                return {"response": response, "route": "rag", "source": "chroma_db"}
            else:
                # RAG falló → Fallback a CAG con contexto completo
                logger.warning("⚠️ RAG no encontró contexto → fallback a CAG con texto completo")
                full_text = rag_engine.get_document_text()
                logger.debug(f"📊 Longitud del texto completo: {len(full_text)} caracteres")
                
                response = cag_engine.generate_response(query, external_context=full_text)
                # Cachear solo si es válida
                self._cache_response_if_valid(query, response)
                return {"response": response, "route": "cag", "source": "full_context"}
                
        except Exception as e:
            logger.error(f"❌ Error en route_query: {str(e)}")
            # Fallback de emergencia (NO cachear errores)
            emergency_response = "Lo siento, ocurrió un error al procesar tu consulta."
            return {"response": emergency_response, "route": "error", "source": "fallback"}

# Instancia singleton para importar
orchestrator = Orchestrator()