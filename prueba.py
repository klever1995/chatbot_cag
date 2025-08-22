# debug_chroma.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from databases.chroma_client import chroma_client

# VERIFICAR si Chroma tiene documentos
print("🔍 Verificando documentos en ChromaDB...")
documentos = chroma_client.collection.get()
print(f"✅ Documentos en Chroma: {len(documentos['ids'])}")

if documentos['ids']:
    print("📄 Primer documento:")
    print(documentos['documents'][0])
else:
    print("❌ CHROMA está VACÍO")