# core/rag_engine.py
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.api.types import Where
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import logging
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# =============================
# Configuración de logging
# =============================
logger = logging.getLogger(__name__)

# =============================
# Cargar variables de entorno
# =============================
load_dotenv()

# =============================
# Cliente Azure OpenAI
# =============================
openai_client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION", "2024-08-01-preview")
)

# =============================
# RAG Engine Mejorado
# =============================
class RAGEngine:
    def __init__(self, chroma_client: chromadb.Client, collection_name: str, openai_client: AzureOpenAI):
        self.chroma_client = chroma_client
        self.collection = chroma_client.get_or_create_collection(
            collection_name, 
            metadata={"hnsw:space": "cosine"}
        )
        self.openai_client = openai_client
        self.documents_store = {}
        
        # Modelo para re-ranking (cross-encoder)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        logger.info("RAG Engine mejorado inicializado con chunking semántico y re-ranking")

    def _semantic_chunking(self, text: str, doc_id: str) -> List[Tuple[str, Dict]]:
        """Divide el texto en chunks semánticos basados en títulos y estructura"""
        chunks = []
        
        # Patrones para identificar títulos y secciones
        title_patterns = [
            r'^(?:ARTÍCULO|Artículo|ART|Art)\s+\d+[.:]\s*(.+)$',
            r'^(\d+\.\d+\.?\s+.+)$',
            r'^([A-Z][A-Z\sáéíóúñÁÉÍÓÚÑ]{10,})$',
            r'^(?:Sección|SECCIÓN|Capítulo|CAPÍTULO)\s+\d+[.:]\s*(.+)$'
        ]
        
        lines = text.split('\n')
        current_chunk = []
        current_title = "Introducción"
        current_metadata = {
            "doc_id": doc_id,
            "title": current_title,
            "chunk_type": "semantic"
        }
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Verificar si la línea es un título
            is_title = False
            matched_title = None
            for pattern in title_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    is_title = True
                    matched_title = match.group(1).strip()
                    break
            
            if is_title and matched_title:
                # Guardar chunk anterior si existe
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    if len(chunk_text) > 50:
                        chunks.append((chunk_text, current_metadata.copy()))
                    
                    current_chunk = []
                
                current_title = matched_title
                current_metadata["title"] = current_title
                current_chunk.append(line)
            else:
                current_chunk.append(line)
                
            # Forzar break cada 400 caracteres aprox
            current_text = '\n'.join(current_chunk)
            if len(current_text) > 600 and not is_title:
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append((chunk_text, current_metadata.copy()))
                    current_chunk = []
        
        # Añadir el último chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text) > 50:
                chunks.append((chunk_text, current_metadata.copy()))
        
        logger.info(f"Chunking semántico: {len(chunks)} chunks para {doc_id}")
        return chunks

    def ingest_document(self, doc_id: str, text: str):
        """Ingesta documento con chunking semántico mejorado"""
        try:
            # Almacenar texto completo
            self.documents_store[doc_id] = text
            logger.info(f"Documento {doc_id} almacenado ({len(text)} caracteres)")

            # Generar chunks semánticos
            semantic_chunks = self._semantic_chunking(text, doc_id)
            
            # Ingresar chunks a ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for idx, (chunk_text, metadata) in enumerate(semantic_chunks):
                chunk_id = f"{doc_id}_chunk_{idx}"
                ids.append(chunk_id)
                documents.append(chunk_text)
                metadatas.append(metadata)
            
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Documento {doc_id} procesado: {len(semantic_chunks)} chunks semánticos")

        except Exception as e:
            logger.error(f"Error ingiriendo documento {doc_id}: {str(e)}")
            raise

    def _hybrid_search(self, query: str, top_k: int = 20) -> List[Tuple[str, Dict]]:
        """Búsqueda híbrida: vector + BM25"""
        try:
            # 1. Búsqueda vectorial
            vector_results = self.collection.query(
                query_texts=[query],
                n_results=top_k * 2,
                include=["metadatas", "documents", "distances"]
            )
            
            # 2. Búsqueda léxica (BM25)
            all_documents = [doc for doc in self.collection.get()["documents"] if doc]
            if all_documents:
                tokenized_corpus = [doc.split() for doc in all_documents]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = query.split()
                bm25_scores = bm25.get_scores(tokenized_query)
                
                # Combinar resultados
                hybrid_results = []
                seen_ids = set()
                
                # Añadir resultados vectoriales
                for i, (doc, metadata, distance) in enumerate(zip(
                    vector_results["documents"][0],
                    vector_results["metadatas"][0],
                    vector_results["distances"][0]
                )):
                    if doc and doc.strip():
                        hybrid_results.append({
                            "text": doc,
                            "metadata": metadata,
                            "score": 1 - distance,
                            "type": "vector"
                        })
                        seen_ids.add(doc[:100])
                
                # Añadir mejores resultados BM25 no duplicados
                bm25_indices = bm25_scores.argsort()[::-1][:top_k]
                for idx in bm25_indices:
                    if idx < len(all_documents) and all_documents[idx]:
                        doc_text = all_documents[idx]
                        if doc_text[:100] not in seen_ids and bm25_scores[idx] > 0:
                            hybrid_results.append({
                                "text": doc_text,
                                "metadata": {"doc_id": "bm25_result", "chunk_type": "bm25"},
                                "score": float(bm25_scores[idx]),
                                "type": "bm25"
                            })
                            seen_ids.add(doc_text[:100])
                
                return hybrid_results
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error en búsqueda híbrida: {str(e)}")
            return []

    def _rerank_results(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """Re-rank resultados usando cross-encoder"""
        if not results:
            return []
        
        try:
            # Preparar pares para re-ranking
            pairs = [(query, result["text"]) for result in results]
            
            # Calcular scores de re-ranking
            rerank_scores = self.cross_encoder.predict(pairs)
            
            # Combinar scores
            for i, result in enumerate(results):
                original_score = result["score"]
                rerank_score = float(rerank_scores[i])
                
                # Combinar scores (50% original, 50% re-rank)
                combined_score = (original_score * 0.5) + (rerank_score * 0.5)
                result["combined_score"] = combined_score
                result["rerank_score"] = rerank_score
            
            # Ordenar por score combinado
            results.sort(key=lambda x: x["combined_score"], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error en re-ranking: {str(e)}")
            return results[:top_k]

    def _build_context(self, query: str, top_k: int = 5) -> str:
        """Construye contexto usando búsqueda híbrida y re-ranking"""
        try:
            # 1. Búsqueda híbrida
            hybrid_results = self._hybrid_search(query, top_k * 3)
            
            if not hybrid_results:
                return ""
            
            # 2. Re-ranking
            reranked_results = self._rerank_results(query, hybrid_results, top_k)
            
            # 3. Construir contexto
            context_parts = []
            for result in reranked_results:
                context_parts.append(f"[Relevancia: {result['combined_score']:.3f}]")
                context_parts.append(result["text"])
                context_parts.append("---")
            
            context = "\n".join(context_parts)
            
            logger.debug(f"Contexto construido: {len(context)} caracteres, {len(reranked_results)} chunks")
            return context
            
        except Exception as e:
            logger.error(f"Error construyendo contexto: {str(e)}")
            return ""

    def generate_answer(self, query: str, top_k: int = 5) -> str:
        """Genera respuesta usando RAG mejorado"""
        try:
            context = self._build_context(query, top_k=top_k)
            
            if not context:
                logger.warning("No se encontró contexto relevante")
                return "No lo sé"
            
            prompt = f"""
Eres un asistente especializado en documentación legal y corporativa. Responde la pregunta del usuario basándote ÚNICAMENTE en el contexto proporcionado.

CONTEXTO:
{context}

INSTRUCCIONES:
1. Responde de manera precisa y concisa usando SOLO la información del contexto
2. Si la respuesta no está en el contexto, di EXACTAMENTE "No lo sé"
3. Cita los artículos o secciones relevantes cuando sea posible
4. Usa el mismo idioma de la pregunta

PREGUNTA: {query}

RESPUESTA:
"""
            response = self.openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
                messages=[
                    {"role": "system", "content": "Eres un experto en análisis documental."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Validar si la respuesta es significativa
            if self._is_negative_response(answer):
                return "No lo sé"
                
            return answer
            
        except Exception as e:
            logger.error(f"Error generando respuesta RAG: {str(e)}")
            return "No lo sé"

    def _is_negative_response(self, response: str) -> bool:
        """Determina si la respuesta es negativa"""
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
            "la información no está disponible"
        ]
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in negative_phrases)

    def get_document_text(self, doc_id: str = None) -> str:
        """Obtiene texto completo de documento(s)"""
        try:
            if doc_id:
                return self.documents_store.get(doc_id, "")
            else:
                return "\n\n".join([f"[DOCUMENTO: {doc_id}]\n{content}" 
                                  for doc_id, content in self.documents_store.items()])
        except Exception as e:
            logger.error(f"Error obteniendo texto completo: {str(e)}")
            return ""

# =============================
# Inicializar cliente Chroma y crear instancia singleton
# =============================
chroma_client = chromadb.Client()
rag_engine = RAGEngine(chroma_client, "docs_collection", openai_client)