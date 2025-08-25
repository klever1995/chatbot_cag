import logging
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MemoryCacheManager:
    """Gestor de caché de respuestas en memoria (reemplazo para Redis)"""
    
    def __init__(self):
        self.cache = {}
        logger.info("✅ Caché en memoria inicializada")
    
    def get_cached_response(self, query):
        """Obtiene una respuesta de la caché si existe y no ha expirado"""
        try:
            cache_key = f"response:{query}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                # Verificar expiración
                if datetime.now() < cached_data["expiration"]:
                    logger.debug(f"✅ Respuesta encontrada en caché para: {query}")
                    return cached_data["response"]
                else:
                    # Eliminar si expiró
                    del self.cache[cache_key]
            return None
        except Exception as e:
            logger.error(f"❌ Error obteniendo de caché: {str(e)}")
            return None
    
    def set_cached_response(self, query, response, expiration_hours=24):
        """Almacena una respuesta en la caché con tiempo de expiración"""
        try:
            cache_key = f"response:{query}"
            expiration_time = datetime.now() + timedelta(hours=expiration_hours)
            
            self.cache[cache_key] = {
                "response": response,
                "expiration": expiration_time
            }
            logger.debug(f"✅ Respuesta almacenada en caché para: {query}")
        except Exception as e:
            logger.error(f"❌ Error almacenando en caché: {str(e)}")
    
    def clear_cache(self):
        """Limpia toda la caché (para desarrollo)"""
        try:
            self.cache.clear()
            logger.warning("🔄 Caché en memoria limpiada")
        except Exception as e:
            logger.error(f"❌ Error limpiando caché: {str(e)}")

# Instancia singleton 
redis_cache = MemoryCacheManager()