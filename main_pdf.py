# ============================
# CONFIGURACIÓN SSL GLOBAL - DESACTIVAR VERIFICACIÓN
# ============================
import os
import warnings

warnings.filterwarnings("ignore", message="Unverified HTTPS request")
os.environ['SSL_CERT_FILE'] = ""
os.environ['REQUESTS_CA_BUNDLE'] = ""
os.environ['CURL_CA_BUNDLE'] = ""
os.environ['PYTHONHTTPSVERIFY'] = "0"
os.environ['NO_PROXY'] = '*'

# ============================
# IMPORTACIONES
# ============================
import logging
import ssl
import urllib3
import requests
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from PyPDF2 import PdfReader
from typing import Optional, List, Dict, Any
import uuid
import time

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
requests_session = requests.Session()
requests_session.verify = False

# ============================
# MÓDULOS PROPIOS
# ============================
from core.orchestrator import orchestrator
from core.rag_engine import rag_engine
from core.cag_engine import cag_engine
from cache.redis_manager import redis_cache

# ============================
# LOGGING
# ============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================
# FASTAPI
# ============================
app = FastAPI(
    title="Sistema RAG + CAG Optimizado",
    version="2.0.0",
    description="Sistema mejorado con chunking semántico, búsqueda híbrida y re-ranking"
)

# ============================
# CONFIGURACIÓN CORS
# ============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:9000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# MODELOS
# ============================
class QueryRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    session_id: Optional[str] = None
    force_route: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    route: str
    source: str
    session_id: str
    processing_time: float
    query_intent: Optional[Dict[str, Any]] = None

class UploadResponse(BaseModel):
    status: str
    message: str
    doc_id: str
    chunks_processed: int

class SystemStatus(BaseModel):
    status: str
    version: str
    redis_connected: bool
    chroma_connected: bool
    documents_loaded: int
    total_chunks: int

class DebugRequest(BaseModel):
    query: str
    route: str

# ============================
# ESTADO GLOBAL
# ============================
class AppState:
    def __init__(self):
        self.start_time = time.time()
        self.total_queries = 0
        self.successful_queries = 0
        self.documents_processed = 0

app_state = AppState()

# ============================
# ENDPOINTS
# ============================
@app.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Sube un PDF y lo indexa en Chroma con chunking semántico mejorado"""
    try:
        # Validar tipo de archivo
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")

        # Crear directorio de uploads si no existe
        upload_dir = Path("uploaded_docs")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        file_id = f"{file.filename.replace('.', '_')}_{uuid.uuid4().hex[:8]}"

        # Guardar archivo
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Extraer texto del PDF
        reader = PdfReader(str(file_path))
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    full_text += f"\n--- Página {page_num + 1} ---\n{page_text}\n"
            except Exception as e:
                logger.warning(f"Error extrayendo página {page_num + 1}: {e}")
                continue

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF")

        # Ingresar documento al RAG engine
        rag_engine.ingest_document(file_id, full_text)
        app_state.documents_processed += 1

        # Obtener estadísticas de chunks
        collection_info = rag_engine.collection.count()
        chunks_count = collection_info if isinstance(collection_info, int) else 0

        logger.info(f"PDF {file.filename} indexado como {file_id}, {chunks_count} chunks")

        return UploadResponse(
            status="success",
            message=f"Documento {file.filename} indexado correctamente con chunking semántico.",
            doc_id=file_id,
            chunks_processed=chunks_count
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error subiendo PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error procesando PDF: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Procesa consultas con el flujo optimizado Redis → RAG → CAG"""
    start_time = time.time()
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
    
    try:
        logger.info(f"Consulta recibida: '{request.query}' - User: {request.user_id} - Session: {session_id}")

        # Contador de consultas
        app_state.total_queries += 1

        # Forzar ruta específica si se solicita (para debugging)
        if request.force_route:
            if request.force_route.lower() == "rag":
                result = orchestrator.force_rag_search(request.query)
            elif request.force_route.lower() == "cag":
                result = orchestrator.force_cag_search(request.query)
            else:
                result = orchestrator.route_query(request.query)
        else:
            # Flujo normal
            result = orchestrator.route_query(request.query)

        # Actualizar estadísticas de éxito
        if not orchestrator._is_negative_response(result.get("response", "")):
            app_state.successful_queries += 1

        processing_time = time.time() - start_time

        logger.info(f"Respuesta generada via {result.get('source', 'unknown')} "
                   f"- Tiempo: {processing_time:.2f}s - Ruta: {result.get('route', 'unknown')}")

        return QueryResponse(
            response=result.get("response", "Error generando respuesta"),
            route=result.get("route", "error"),
            source=result.get("source", "unknown"),
            session_id=session_id,
            processing_time=processing_time,
            query_intent=result.get("query_intent")
        )

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error procesando consulta: {e}")
        
        return QueryResponse(
            response="Lo siento, ocurrió un error interno al procesar tu consulta.",
            route="error",
            source="system_error",
            session_id=session_id,
            processing_time=processing_time
        )

@app.post("/debug/query")
async def debug_query(request: DebugRequest):
    """Endpoint para debugging de consultas específicas"""
    try:
        if request.route.lower() == "rag":
            result = orchestrator.force_rag_search(request.query)
        elif request.route.lower() == "cag":
            result = orchestrator.force_cag_search(request.query)
        else:
            result = orchestrator.route_query(request.query)

        return JSONResponse({
            "result": result,
            "query": request.query,
            "forced_route": request.route
        })

    except Exception as e:
        logger.error(f"Error en debug query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=SystemStatus)
async def system_status():
    """Retorna el estado del sistema y estadísticas"""
    try:
        # Verificar conexiones
        redis_connected = redis_cache.ping()
        
        # Contar documentos y chunks
        documents_loaded = len(rag_engine.documents_store)
        collection_count = rag_engine.collection.count()
        
        # Calcular uptime
        uptime = time.time() - app_state.start_time

        return SystemStatus(
            status="healthy",
            version="2.0.0",
            redis_connected=redis_connected,
            chroma_connected=collection_count >= 0,
            documents_loaded=documents_loaded,
            total_chunks=collection_count
        )

    except Exception as e:
        logger.error(f"Error obteniendo estado del sistema: {e}")
        return SystemStatus(
            status="degraded",
            version="2.0.0",
            redis_connected=False,
            chroma_connected=False,
            documents_loaded=0,
            total_chunks=0
        )

@app.get("/stats")
async def system_stats():
    """Estadísticas detalladas del sistema"""
    return {
        "uptime_seconds": time.time() - app_state.start_time,
        "total_queries": app_state.total_queries,
        "successful_queries": app_state.successful_queries,
        "success_rate": (app_state.successful_queries / app_state.total_queries * 100) if app_state.total_queries > 0 else 0,
        "documents_processed": app_state.documents_processed,
        "redis_status": "connected" if redis_cache.ping() else "disconnected",
        "version": "2.0.0-optimized"
    }

@app.get("/documents")
async def list_documents():
    """Lista todos los documentos cargados en el sistema"""
    try:
        documents = []
        for doc_id, content in rag_engine.documents_store.items():
            documents.append({
                "doc_id": doc_id,
                "content_length": len(content),
                "first_100_chars": content[:100] + "..." if len(content) > 100 else content
            })
        
        return {"documents": documents, "count": len(documents)}
    
    except Exception as e:
        logger.error(f"Error listando documentos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cache")
async def clear_cache():
    """Limpia toda la cache de Redis"""
    try:
        cleared = redis_cache.clear_all()
        return {"status": "success", "message": f"Cache limpiada, {cleared} items removidos"}
    except Exception as e:
        logger.error(f"Error limpiando cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check simple"""
    return {"status": "healthy", "service": "rag-cag-optimized-system"}

@app.get("/")
async def root():
    return {
        "message": "Sistema RAG + CAG Optimizado funcionando",
        "version": "2.0.0",
        "features": [
            "Chunking semántico mejorado",
            "Búsqueda híbrida (vector + BM25)",
            "Re-ranking con cross-encoder",
            "Optimización de contexto CAG",
            "Sistema de caching inteligente"
        ],
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "query": "/query (POST)",
            "upload": "/upload_pdf (POST)",
            "debug": "/debug/query (POST)",
            "stats": "/stats",
            "documents": "/documents",
            "clear_cache": "/cache (DELETE)"
        }
    }

# ============================
# MANEJO DE ERRORES GLOBAL
# ============================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error global no manejado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor", "error": str(exc)}
    )

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    import uvicorn
    import logging

    # Configurar logging para que solo muestre warnings y errores
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s"  # Esto elimina rutas y líneas
    )

    logger.warning("Iniciando servidor RAG + CAG optimizado...")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
        log_level="warning",   
        access_log=False       
    )