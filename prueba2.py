# prueba_embeddings.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.embeddings import embedding_model

def main():
    print("🧪 Probando modelo de embeddings...")
    
    # Texto de prueba
    test_text = "El hombre llegó a la luna en 1969"
    
    try:
        # Generar embedding
        embedding = embedding_model.get_embeddings(test_text)
        
        print(f"✅ Éxito! Embedding generado")
        print(f"Texto: '{test_text}'")
        print(f"Dimensión del vector: {len(embedding[0])}")
        print(f"Primeros 5 valores: {embedding[0][:5]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()