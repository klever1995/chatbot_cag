# prueba_orchestrator.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import orchestrator

def test_query(query, expected_route):
    """Función auxiliar para probar consultas"""
    actual_route = orchestrator.route_query(query)
    status = "✅" if actual_route == expected_route else "❌"
    print(f"{status} Query: '{query}'")
    print(f"   Esperado: {expected_route.upper()}, Obtenido: {actual_route.upper()}")
    return actual_route == expected_route

def main():
    print("🧪 Probando orquestador inteligente...")
    print("=" * 50)
    
    # Agregar un documento a la caché de contexto para pruebas
    from cache.context_cache import context_cache
    context_cache.add_document("Python es un lenguaje de programación interpretado de alto nivel.")
    
    test_cases = [
        # (consulta, ruta_esperada)
        ("qué es python", "cag"),           # Simple - caché contexto
        ("python", "cag"),                  # Muy corta - CAG
        ("cómo funciona el protocolo HTTP", "cag"),  # Patrón simple - CAG
        ("ventajas y desventajas de python vs java", "rag"),  # Compleja - RAG
        ("diferencia entre listas y tuplas en python", "rag"), # Compleja - RAG
        ("explica el machine learning", "cag"),      # Simple - CAG
        ("qué pasa si mezclamos ácido y base", "rag"), # Compleja - RAG
        ("definición de algoritmo", "cag"), # Simple - CAG
    ]
    
    print("🧠 Probando decisiones de ruteo...")
    print("-" * 30)
    
    passed = 0
    for query, expected_route in test_cases:
        if test_query(query, expected_route):
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Resultado: {passed}/{len(test_cases)} pruebas exitosas")
    
    if passed == len(test_cases):
        print("🎉 ¡Todos los tests pasaron! El orquestador funciona correctamente.")
    else:
        print("⚠️  Algunos tests fallaron. Revisa las reglas heurísticas.")

if __name__ == "__main__":
    main()