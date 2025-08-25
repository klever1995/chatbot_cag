# cache/context_cache.py
import logging
import hashlib
from datetime import datetime, timedelta
import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class ContextCache:
    """Caché de contexto semántica para documentos legales/laborales usando ChromaDB"""
    
    def __init__(self):
        try:
            # Inicializar ChromaDB para búsqueda semántica
            self.client = chromadb.PersistentClient(path="./context_cache_db")
            self.collection = self.client.get_or_create_collection(
                name="legal_context_cache",
                metadata={"hnsw:space": "cosine", "description": "Caché de contexto para documentos legales"}
            )
            
            # Modelo de embeddings para búsqueda semántica
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Caché de contexto semántica inicializada con ChromaDB")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando caché de contexto: {str(e)}")
            raise
    
    def _generate_hash(self, content):
        """Genera un hash único para el contenido del documento"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_document(self, document_content, doc_type="legal", source="uploaded"):
        """Añade un documento a la caché de contexto con embeddings semánticos"""
        try:
            if not document_content or len(document_content.strip()) < 50:
                logger.warning("⚠️ Contenido muy corto para añadir a caché")
                return None
            
            doc_hash = self._generate_hash(document_content)
            
            # Generar embedding semántico
            embedding = self.embedding_model.encode(document_content).tolist()
            
            metadata = {
                "doc_type": doc_type,
                "source": source,
                "added_at": datetime.now().isoformat(),
                "length": len(document_content),
                "hash": doc_hash
            }
            
            # Añadir a ChromaDB con embedding
            self.collection.add(
                ids=[doc_hash],
                documents=[document_content],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            
            logger.debug(f"✅ Documento añadido a caché semántica: {doc_hash[:8]}...")
            return doc_hash
            
        except Exception as e:
            logger.error(f"❌ Error añadiendo documento a caché semántica: {str(e)}")
            return None
    
    def search_in_cache(self, query, top_k=3, similarity_threshold=0.6):
        """
        Busca documentos en la caché usando búsqueda semántica
        Returns:
            list: Documentos relevantes ordenados por similitud
        """
        try:
            if not query or len(query.strip()) < 3:
                return []
            
            # Generar embedding de la consulta
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Buscar en ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Filtrar por threshold de similitud
            relevant_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0], 
                results["metadatas"][0], 
                results["distances"][0]
            )):
                similarity = 1 - distance  # Convertir distancia a similitud
                
                if similarity >= similarity_threshold:
                    relevant_results.append({
                        "document": doc,
                        "metadata": metadata,
                        "similarity": round(similarity, 3),
                        "rank": i + 1
                    })
            
            logger.debug(f"✅ Búsqueda semántica: {len(relevant_results)} resultados relevantes")
            return relevant_results
            
        except Exception as e:
            logger.error(f"❌ Error en búsqueda semántica: {str(e)}")
            return []
    
    def get_context_stats(self):
        """Obtiene estadísticas de la caché de contexto"""
        try:
            stats = self.collection.count()
            return {
                "total_documents": stats,
                "status": "active"
            }
        except Exception as e:
            logger.error(f"❌ Error obteniendo estadísticas: {str(e)}")
            return {"total_documents": 0, "status": "error"}
    
    def clear_old_entries(self, max_age_days=30):
        """Limpia entradas antiguas de la caché (implementación básica)"""
        try:
            count = self.collection.count()
            logger.info(f"🔄 Caché contiene {count} documentos (limpieza manual requerida)")
            return count
            
        except Exception as e:
            logger.error(f"❌ Error en limpieza de caché: {str(e)}")
            return 0
    
    def get_document_by_hash(self, doc_hash):
        """Recupera un documento específico por su hash"""
        try:
            results = self.collection.get(ids=[doc_hash])
            if results["documents"]:
                return {
                    "document": results["documents"][0],
                    "metadata": results["metadatas"][0]
                }
            return None
        except Exception as e:
            logger.error(f"❌ Error recuperando documento: {str(e)}")
            return None

# Instancia singleton para ser importada
context_cache = ContextCache()