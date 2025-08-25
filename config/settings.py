import os
from dotenv import load_dotenv

load_dotenv()

# Configuración de Azure OpenAI 
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")                
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-08-01-preview")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

if not OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
    raise ValueError("❌ OPENAI_API_KEY y AZURE_OPENAI_ENDPOINT son requeridos en .env")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
RAG_THRESHOLD = float(os.getenv("RAG_THRESHOLD", 0.7))