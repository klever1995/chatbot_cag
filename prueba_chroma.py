# prueba_chroma.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from databases.chroma_client import chroma_client

def main():
    print("🧪 Probando conexión con ChromaDB...")
    
    # Documentos de prueba
    test_documents = [
        "El hombre llegó a la luna en 1969",
        "La inteligencia artificial está transformando el mundo",
        "Python es un lenguaje de programación muy popular"
    ]
    
    try:
        # Añadir documentos a la colección
        print("📝 Añadiendo documentos de prueba...")
        chroma_client.add_documents(test_documents)
        
        # Buscar documentos similares
        print("🔍 Buscando documentos similares...")
        query = "viaje espacial a la luna"
        results = chroma_client.search(query, n_results=2)
        
        print(f"✅ Búsqueda exitosa para: '{query}'")
        print(f"📄 Documentos encontrados: {len(results['documents'][0])}")
        
        for i, doc in enumerate(results['documents'][0]):
            print(f"{i+1}. {doc}")
            
        # Opcional: resetear la colección después de la prueba
        # chroma_client.reset_collection()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()