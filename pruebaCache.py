# prueba_context_cache.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cache.context_cache import context_cache

def main():
    print("🧪 Probando caché de contexto...")
    
    # Documentos de prueba
    test_documents = [
        "El protocolo HTTP es fundamental para la web",
        "Python es un lenguaje de programación interpretado",
        "Los modelos de machine learning requieren datos de entrenamiento"
    ]
    
    try:
        # Añadir documentos a la caché
        print("📝 Añadiendo documentos a la caché...")
        for doc in test_documents:
            context_cache.add_document(doc, {"type": "technical"})
        
        # Buscar documento específico
        print("🔍 Buscando documento existente...")
        found_doc = context_cache.get_document(test_documents[0])
        
        if found_doc:
            print(f"✅ Documento encontrado: {found_doc['content'][:50]}...")
        else:
            print("❌ Documento no encontrado")
        
        # Buscar por similitud (búsqueda simple)
        print("🔎 Buscando por similitud...")
        results = context_cache.search_in_cache("lenguaje programación Python")
        
        print(f"✅ Resultados de búsqueda: {len(results)} encontrados")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['document'][:60]}...")
            
        # Probar con documento no existente
        print("🔍 Buscando documento no existente...")
        missing_doc = context_cache.get_document("Este documento no existe")
        if not missing_doc:
            print("✅ Comportamiento correcto: documento no encontrado")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()