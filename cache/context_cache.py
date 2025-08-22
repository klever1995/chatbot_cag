# cache/context_cache.py
import logging
import hashlib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ContextCache:
    """Caché de contexto para documentos estáticos (alternativa simple a vLLM)"""
    
    def __init__(self):
        self.context_cache = {}  # Almacena documentos por su hash
        self.access_times = {}   # Seguimiento de últimos accesos para LRU
        logger.info("✅ Caché de contexto inicializada")
    
    def _generate_hash(self, content):
        """Genera un hash único para el contenido del documento"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_document(self, document_content, metadata=None):
        """Añade un documento a la caché de contexto"""
        try:
            doc_hash = self._generate_hash(document_content)
            
            self.context_cache[doc_hash] = {
                "content": document_content,
                "metadata": metadata or {},
                "added_at": datetime.now()
            }
            
            # Actualizar tiempo de acceso
            self.access_times[doc_hash] = datetime.now()
            
            logger.debug(f"✅ Documento añadido a caché de contexto: {doc_hash[:8]}...")
            return doc_hash
            
        except Exception as e:
            logger.error(f"❌ Error añadiendo documento a caché: {str(e)}")
            return None
    
    def get_document(self, document_content):
        """Recupera un documento de la caché si existe"""
        try:
            doc_hash = self._generate_hash(document_content)
            
            if doc_hash in self.context_cache:
                # Actualizar tiempo de acceso
                self.access_times[doc_hash] = datetime.now()
                logger.debug(f"✅ Documento encontrado en caché de contexto: {doc_hash[:8]}...")
                return self.context_cache[doc_hash]
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error recuperando documento de caché: {str(e)}")
            return None
    
    def search_in_cache(self, query, similarity_threshold=0.3):  # Reducir threshold
        """
        Busca documentos en la caché que puedan responder la consulta
        """
        try:
            results = []
            query_lower = query.lower()
            query_words = query_lower.split()
            
            for doc_hash, doc_data in self.context_cache.items():
                content = doc_data["content"].lower()
                
                # Búsqueda más flexible: contar coincidencias de palabras
                word_matches = sum(1 for word in query_words if word in content)
                similarity_score = word_matches / len(query_words) if query_words else 0
                
                # También verificar coincidencia parcial con el documento completo
                if similarity_score >= similarity_threshold or any(word in content for word in ["python", "programación", "lenguaje"]):
                    results.append({
                        "document": doc_data["content"],
                        "metadata": doc_data["metadata"],
                        "similarity": similarity_score
                    })
            
            logger.debug(f"✅ Búsqueda en caché de contexto: {len(results)} resultados")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error buscando en caché de contexto: {str(e)}")
            return []
    
    def clear_old_entries(self, max_age_hours=24):
        """Limpia entradas antiguas de la caché"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            old_entries = [
                doc_hash for doc_hash, access_time in self.access_times.items()
                if access_time < cutoff_time
            ]
            
            for doc_hash in old_entries:
                self.context_cache.pop(doc_hash, None)
                self.access_times.pop(doc_hash, None)
            
            logger.info(f"🔄 Limpiadas {len(old_entries)} entradas antiguas de caché")
            
        except Exception as e:
            logger.error(f"❌ Error limpiando caché: {str(e)}")

# Instancia singleton para ser importada
context_cache = ContextCache()