import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cache.redis_manager import redis_cache

def main():
    print("🧪 Probando caché en memoria...")

    # Datos de prueba
    test_query = "¿Cuándo llegó el hombre a la luna?"
    test_response = "El hombre llegó a la luna en 1969"

    try:
        # Guardar en caché
        print("💾 Guardando respuesta en caché...")
        redis_cache.set_cached_response(test_query, test_response)

        # Recuperar de caché
        print("🔍 Buscando en caché...")
        cached_response = redis_cache.get_cached_response(test_query)

        if cached_response:
            print(f"✅ Respuesta recuperada de caché: '{cached_response}'")
        else:
            print("❌ No se encontró en caché")

        # Limpiar caché (opcional)
        redis_cache.clear_cache()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
