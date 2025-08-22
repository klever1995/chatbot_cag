# prueba_rag.py
import sys
import os
os.environ['NO_PROXY'] = 'openai.azure.com,azure.com,recursoazureopenaimupi.openai.azure.com'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.rag_engine import rag_engine
from databases.chroma_client import chroma_client

def main():
    print("🧪 Probando motor RAG...")
    print("=" * 50)
    
    # Limpiar colección primero
    print("🔄 Limpiando colección...")
    chroma_client.reset_collection()
    
    # Documentos de prueba
    test_documents = [
        "Python es un lenguaje de programación interpretado de alto nivel creado por Guido van Rossum en 1991.",
        "Los lenguajes interpretados como Python no necesitan compilación previa, se ejecutan línea por línea.",
        "JavaScript es otro lenguaje interpretado principalmente usado para desarrollo web frontend.",
        "Los lenguajes compilados como C++ requieren compilación antes de su ejecución.",
        "Python es conocido por su sintaxis clara y legible que favorece la productividad del desarrollador."
    ]
    
    print("📝 Añadiendo documentos de prueba a ChromaDB...")
    chroma_client.add_documents(test_documents)
    
    # Consultas de prueba
    test_queries = [
        "¿Qué es Python y quién lo creó?",
        "¿Cuál es la diferencia entre lenguajes interpretados y compilados?",
        "¿Para qué se usa principalmente JavaScript?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Consulta {i}: '{query}'")
        print("-" * 40)
        
        try:
            response = rag_engine.generate_response(query, n_results=3)
            print(f"✅ Respuesta:\n{response}")
            
        except Exception as e:
            print(f"❌ Error generando respuesta: {e}")
    
    print("\n✅ Prueba de RAG completada")

if __name__ == "__main__":
    main()