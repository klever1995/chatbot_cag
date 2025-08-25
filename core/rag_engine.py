from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import os
import logging
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuración de logging
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Cliente Azure OpenAI con retry
openai_client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION", "2024-08-01-preview")
)

# RAG Engine Optimizado
class RAGEngine:
    def __init__(self, chroma_client: chromadb.Client, collection_name: str, openai_client: AzureOpenAI):
        self.chroma_client = chroma_client
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} 
        )
        self.openai_client = openai_client
        logger.info(f"✅ RAG Engine inicializado para colección: {collection_name}")

    # Ingestar documentos con chunking jerárquico y manejo de errores
    def ingest_document(self, doc_id: str, text: str):
        """Ingesta un documento con chunking jerárquico y manejo robusto de errores"""
        try:
            if not text or text.strip() == "":
                logger.warning(f"⚠️ Documento {doc_id} vacío, omitiendo ingesta")
                return

            # Limpiar documentos existentes del mismo ID para evitar duplicados
            self.collection.delete(where={"doc_id": doc_id})
            
            large_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]  
            )
            large_chunks = large_splitter.split_text(text)

            logger.info(f"📄 Procesando documento {doc_id} con {len(large_chunks)} chunks grandes")

            for section_id, large_chunk in enumerate(large_chunks):
                # Ingestar chunk grande
                self.collection.add(
                    ids=[f"{doc_id}_L{section_id}"],
                    documents=[large_chunk],
                    metadatas=[{
                        "doc_id": doc_id,
                        "section_id": section_id,
                        "chunk_id": -1,
                        "level": "large",
                        "chunk_type": "section"
                    }]
                )

                # Crear chunks pequeños del chunk grande
                small_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=300, 
                    chunk_overlap=50,
                    separators=["\n", ". ", "! ", "? ", " ", ""]
                )
                small_chunks = small_splitter.split_text(large_chunk)
                
                for chunk_id, small_chunk in enumerate(small_chunks):
                    # Ignorar chunks vacíos
                    if small_chunk.strip():  
                        self.collection.add(
                            ids=[f"{doc_id}_L{section_id}_S{chunk_id}"],
                            documents=[small_chunk],
                            metadatas=[{
                                "doc_id": doc_id,
                                "section_id": section_id,
                                "chunk_id": chunk_id,
                                "level": "small",
                                "chunk_type": "detail"
                            }]
                        )

            logger.info(f"✅ Documento {doc_id} indexado exitosamente con {len(large_chunks)} secciones")

        except Exception as e:
            logger.error(f"❌ Error ingiriendo documento {doc_id}: {str(e)}")
            raise

    # Recuperación jerárquica optimizada
    def _build_context(self, query: str, top_k: int = 7) -> str:
        """Construye contexto mediante recuperación jerárquica con manejo de errores"""
        try:
            #  Buscar chunks pequeños (detalles específicos)
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"level": "small"},
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["documents"] or not results["documents"][0]:
                return "No se encontró información relevante en los documentos."

            retrieved_chunks = []
            seen_large_sections = set()

            # Procesar chunks pequeños encontrados
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0], 
                results["metadatas"][0], 
                results["distances"][0]
            )):
                similarity_score = 1 - distance  
                retrieved_chunks.append(f"📝 Detalle [{similarity_score:.0%}]: {doc}")
                
                # Obtener el chunk grande padre para contexto
                section_id = metadata["section_id"]
                doc_id = metadata["doc_id"]
                section_key = (doc_id, section_id)
                
                if section_key not in seen_large_sections:
                    try:
                        parent_results = self.collection.query(
                            query_texts=[query],
                            n_results=1,
                            where={"$and": [
                                {"doc_id": doc_id}, 
                                {"section_id": section_id}, 
                                {"level": "large"}
                            ]},
                            include=["documents", "distances"]
                        )
                        
                        if parent_results["documents"] and parent_results["documents"][0]:
                            parent_text = parent_results["documents"][0][0]
                            parent_similarity = 1 - parent_results["distances"][0][0] if parent_results["distances"][0] else 0
                            retrieved_chunks.append(f"📄 Contexto [{parent_similarity:.0%}]: {parent_text}")
                            seen_large_sections.add(section_key)
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Error recuperando chunk grande: {str(e)}")
                        continue

            # Limitar el contexto para no exceder límites de tokens
            max_context_length = 6000
            context = "\n\n".join(retrieved_chunks)
            if len(context) > max_context_length:
                context = context[:max_context_length] + "... [contexto truncado]"
                
            return context

        except Exception as e:
            logger.error(f"❌ Error construyendo contexto: {str(e)}")
            return "Error recuperando información de los documentos."

    # Generación de respuesta 
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _generate_with_retry(self, messages: List[Dict]) -> str:
        """Genera respuesta con retry para manejar errores de API"""
        try:
            response = self.openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
                messages=messages,
                temperature=0.1, 
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Error llamando a Azure OpenAI: {str(e)}")
            raise

    def generate_answer(self, query: str, top_k: int = 7) -> str:
        """Genera respuesta usando RAG con manejo robusto de errores"""
        try:
            logger.info(f"🔍 Buscando respuesta para: '{query}'")
            
            context = self._build_context(query, top_k=top_k)
            
            if "No se encontró información" in context or "Error recuperando" in context:
                return context

            prompt = f"""
Eres un experto asistente legal y laboral. Responde la pregunta del usuario basándote ÚNICAMENTE en el contexto proporcionado.

REGLAS ESTRICTAS:
1. Responde SOLO con información del contexto proporcionado
2. Si la información no está en el contexto, di: "No tengo información sobre esto en los documentos disponibles"
3. Sé preciso, conciso y profesional
4. Mantén el mismo idioma de la pregunta
5. Considera el score de similitud al evaluar la relevancia de la información

CONTEXTO DE LOS DOCUMENTOS (con scores de relevancia):
{context}

PREGUNTA DEL USUARIO: {query}

RESPUESTA:
"""

            messages = [
                {
                    "role": "system", 
                    "content": "Eres un asistente especializado en documentación legal y laboral. Responde de manera precisa y profesional."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]

            response = self._generate_with_retry(messages)
            logger.info("✅ Respuesta RAG generada exitosamente")
            return response

        except Exception as e:
            logger.error(f"❌ Error generando respuesta RAG: {str(e)}")
            return "Lo siento, ocurrió un error al procesar tu consulta. Por favor, intenta nuevamente."

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

# Inicializar cliente Chroma y crear instancia singleton
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")  
    rag_engine = RAGEngine(chroma_client, "legal_docs_collection", openai_client)
    logger.info("✅ Cliente ChromaDB inicializado en modo persistente")
except Exception as e:
    logger.error(f"❌ Error inicializando ChromaDB: {str(e)}")
    # Fallback a cliente en memoria
    chroma_client = chromadb.Client()
    rag_engine = RAGEngine(chroma_client, "legal_docs_collection", openai_client)
    logger.info("✅ Cliente ChromaDB inicializado en modo memoria")