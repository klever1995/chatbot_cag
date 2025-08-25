import logging
import chromadb
from chromadb.config import Settings
import os

logger = logging.getLogger(__name__)

class ChromaClient:
    """Cliente para interactuar con ChromaDB local"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        try:
            # Configurar cliente de Chroma local con persistencia
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False 
                )
            )
            
            # Crear o obtener la colección
            self.collection = self.client.get_or_create_collection(
                name="legal_documents",
                metadata={
                    "hnsw:space": "cosine",
                    "description": "Colección de documentos legales y laborales"
                }
            )
            
            logger.info(f"✅ Cliente ChromaDB local inicializado en: {persist_directory}")
            logger.info(f"✅ Colección 'legal_documents' lista (documentos: {self.collection.count()})")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando ChromaDB: {str(e)}")
            # Fallback a cliente en memoria
            try:
                self.client = chromadb.Client()
                self.collection = self.client.get_or_create_collection(
                    name="legal_documents",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.warning("🔄 Usando ChromaDB en memoria por error de persistencia")
            except Exception as fallback_error:
                logger.critical(f"💥 Error crítico con ChromaDB: {fallback_error}")
                raise
    
    def add_documents(self, documents, ids=None, metadatas=None):
        """Añade documentos a la colección (ahora usa embeddings de Chroma por defecto)"""
        if not documents:
            logger.warning("⚠️ Intento de añadir documentos vacíos")
            return
        
        try:
            # Si no se proporcionan IDs, generar automáticamente
            if ids is None:
                ids = [f"doc_{i}_{hash(doc[:50])}" for i, doc in enumerate(documents)]
            
            # Añadir a la colección
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            
            logger.info(f"✅ Añadidos {len(documents)} documentos a la colección")
            
        except Exception as e:
            logger.error(f"❌ Error añadiendo documentos: {str(e)}")
            raise
    
    def search(self, query_text, n_results=5, where_filter=None):
        """Busca documentos similares a la consulta usando embeddings de Chroma"""
        try:
            # Realizar búsqueda semántica 
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            logger.debug(f"✅ Búsqueda completada: {len(results['documents'][0])} resultados")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en búsqueda semántica: {str(e)}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'ids': [[]]}
    
    def search_by_embedding(self, query_embedding, n_results=5, where_filter=None):
        """Búsqueda usando embedding precalculado (para integración avanzada)"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            return results
        except Exception as e:
            logger.error(f"❌ Error en búsqueda por embedding: {str(e)}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'ids': [[]]}
    
    def get_collection_stats(self):
        """Obtiene estadísticas de la colección"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name,
                "status": "active"
            }
        except Exception as e:
            logger.error(f"❌ Error obteniendo estadísticas: {str(e)}")
            return {"total_documents": 0, "status": "error"}
    
    def reset_collection(self):
        """Elimina todos los documentos de la colección (para desarrollo)"""
        try:
            count_before = self.collection.count()
            self.client.delete_collection(name="legal_documents")
            self.collection = self.client.get_or_create_collection(
                name="legal_documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.warning(f"🔄 Colección resetada: {count_before} documentos eliminados")
            return count_before
        except Exception as e:
            logger.error(f"❌ Error reseteando colección: {str(e)}")
            try:

                existing_docs = self.collection.get()
                if existing_docs['ids']:
                    self.collection.delete(ids=existing_docs['ids'])
                    logger.warning(f"🔄 Colección limpiada: {len(existing_docs['ids'])} documentos eliminados")
                    return len(existing_docs['ids'])
            except Exception as fallback_error:
                logger.error(f"❌ Error en método alternativo de reset: {fallback_error}")
                raise
    
    def get_document(self, document_id):
        """Obtiene un documento específico por ID"""
        try:
            results = self.collection.get(ids=[document_id])
            if results['documents']:
                return {
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0] if results['metadatas'] else {}
                }
            return None
        except Exception as e:
            logger.error(f"❌ Error obteniendo documento {document_id}: {str(e)}")
            return None

# Instancia singleton para ser importada
chroma_client = ChromaClient()