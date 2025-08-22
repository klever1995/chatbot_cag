# main.py
import os
os.environ['NO_PROXY'] = 'recursoazureopenaimupi.openai.azure.com'
import logging
import ssl
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.orchestrator import orchestrator
from core.rag_engine import rag_engine
from core.cag_engine import cag_engine
from evaluation.langfuse_monitoring import langfuse_monitor

# ============================
# CONFIGURACIÓN SSL GLOBAL
# ============================
# Ruta al certificado CA local
ca_cert_path = os.path.join(os.path.dirname(__file__), "ca-cert.pem")

# Establecer variable de entorno para todo Python
os.environ["SSL_CERT_FILE"] = ca_cert_path

# Crear sesión de requests global con el certificado
requests_session = requests.Session()
requests_session.verify = ca_cert_path

# ============================
# CONFIGURAR LOGGING
# ============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar NO_PROXY para Azure OpenAI
os.environ['NO_PROXY'] = 'openai.azure.com,azure.com,recursoazureopenaimupi.openai.azure.com'

# ============================
# INICIALIZAR FASTAPI
# ============================
app = FastAPI(title="Sistema RAG + CAG", version="1.0.0")

# ============================
# MODELOS Pydantic
# ============================
class QueryRequest(BaseModel):
    query: str
    user_id: str = "anonymous"

class QueryResponse(BaseModel):
    response: str
    route: str
    source: str

# ============================
# ENDPOINTS
# ============================
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Endpoint principal para procesar consultas"""
    try:
        logger.info(f"📥 Consulta recibida: '{request.query}'")
        
        # 1. Decidir la ruta (RAG vs CAG)
        route = orchestrator.route_query(request.query)
        logger.info(f"🔄 Ruta decidida: {route.upper()}")
        
        # 2. Procesar según la ruta
        if route == "rag":
            response = rag_engine.process_query(request.query)
            source = "chroma_db"
        else:
            response = cag_engine.process_query(request.query)
            source = "context_cache"
        
        # 3. Monitorear con Langfuse (opcional)
        try:
            langfuse_monitor.trace_query(
                query=request.query,
                response=response,
                route=route,
                user_id=request.user_id
            )
        except Exception as e:
            logger.warning(f"⚠️ Error en monitoreo Langfuse: {e}")
        
        logger.info(f"📤 Respuesta generada via {route.upper()}")
        
        return QueryResponse(
            response=response,
            route=route,
            source=source
        )
        
    except Exception as e:
        logger.error(f"❌ Error procesando consulta: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/health")
async def health_check():
    """Endpoint de health check"""
    return {"status": "healthy", "service": "rag-cag-system"}

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "Sistema RAG + CAG funcionando",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)"
        }
    }

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
