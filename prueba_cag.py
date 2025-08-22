# prueba_cag.py
import sys
import os
os.environ['NO_PROXY'] = 'openai.azure.com,azure.com,recursoazureopenaimupi.openai.azure.com'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.cag_engine import cag_engine
from cache.context_cache import context_cache

def main():
    print("🧪 Probando motor CAG...")
    print("=" * 50)
    
    # Limpiar caché de contexto primero
    print("🔄 Limpiando caché de contexto...")
    # No hay método clear, pero añadiremos documentos frescos
    
    # Documentos de prueba para caché de contexto
    print("📝 Añadiendo documentos a caché de contexto...")
    test_documents = [
        "Python es un lenguaje de programación interpretado de alto nivel creado por Guido van Rossum en 1991.",
        "Los lenguajes interpretados como Python no necesitan compilación previa, se ejecutan línea por línea.",
        "JavaScript es otro lenguaje interpretado principalmente usado para desarrollo web frontend.",
        "Los lenguajes compilados como C++ requieren compilación antes de su ejecución.",
        "Python es conocido por su sintaxis clara y legible que favorece la productividad del desarrollador."
    ]
    
    for doc in test_documents:
        context_cache.add_document(doc, {"type": "programming"})
    
    # Consultas de prueba que deberían ser respondidas desde CAG
    test_queries = [
        "qué es python",
        "quien creó python",
        "cómo funcionan los lenguajes interpretados",
        "para qué se usa javascript"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Consulta {i}: '{query}'")
        print("-" * 40)
        
        try:
            response = cag_engine.generate_response(query)
            print(f"✅ Respuesta CAG:\n{response}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ Prueba de CAG completada")

if __name__ == "__main__":
    main()