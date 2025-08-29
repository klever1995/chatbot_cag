# core/orchestrator.py
import logging
import re
from cache.redis_manager import redis_cache
from core.rag_engine import rag_engine
from core.cag_engine import cag_engine
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Orchestrator:
    """Orquestador optimizado con flujo Redis → RAG → CAG"""

    def __init__(self):
        logger.info("Orquestador optimizado inicializado")

    def _should_fallback_to_cag(self, response: str) -> bool:
        """Determina si la respuesta del RAG requiere fallback a CAG"""
        if not response or not response.strip():
            return True
        
        negative_indicators = [
            "no lo sé",
            "no encuentro",
            "no tengo información",
            "no está en el contexto",
            "no aparece en el texto",
            "no se menciona",
            "no hay información",
            "no puedo responder",
            "la información no está disponible"
        ]
        
        response_lower = response.lower().strip()
        
        # Caso especial: respuesta exacta "No lo sé"
        if response_lower == "no lo sé":
            return True
        
        # Verificar múltiples indicadores negativos
        negative_count = sum(1 for indicator in negative_indicators if indicator in response_lower)
        
        # Si más de 1 indicador negativo o respuesta muy corta sin contenido
        if negative_count >= 1 or (len(response_lower) < 20 and any(indicator in response_lower for indicator in negative_indicators)):
            return True
            
        return False

    def _is_negative_response(self, response: str) -> bool:
        """Determina si la respuesta es negativa (no debe cachearse)"""
        if not response or not response.strip():
            return True
            
        negative_phrases = [
            "no lo sé",
            "no tengo información",
            "no está en el contexto",
            "no encuentro",
            "no se menciona",
            "no aparece",
            "no puedo responder",
            "no hay información",
            "la información no está disponible",
            "no tengo datos"
        ]
        
        response_lower = response.lower().strip()
        
        # Respuesta exacta "No lo sé"
        if response_lower == "no lo sé":
            return True
            
        # Verificar frases negativas con contexto
        has_negative = any(phrase in response_lower for phrase in negative_phrases)
        
        # Si es una respuesta muy corta y negativa
        if has_negative and len(response_lower) < 50:
            return True
            
        return False

    def _cache_response_if_valid(self, query: str, response: str, route: str) -> None:
        """Cachea la respuesta solo si no es negativa y viene de RAG o CAG exitoso"""
        try:
            if not self._is_negative_response(response):
                # Solo cachear respuestas positivas de RAG o CAG con contexto válido
                if route in ["rag", "cag"]:
                    redis_cache.set_cached_response(query, response)
                    logger.debug(f"Respuesta válida cacheada en Redis desde {route}")
                else:
                    logger.debug(f"Respuesta de {route} no cacheada")
            else:
                logger.debug("Respuesta negativa NO cacheada en Redis")
                
        except Exception as e:
            logger.error(f"Error cacheando respuesta: {str(e)}")

    def _extract_query_intent(self, query: str) -> Dict[str, Any]:
        """Analiza la intención de la consulta para optimizar el routing"""
        query_lower = query.lower()
        
        # Palabras clave que sugieren búsqueda específica en documentos
        document_keywords = [
            "artículo", "art", "sección", "capítulo", "prohibición",
            "norma", "reglamento", "política", "contrato", "cláusula",
            "documento", "pdf", "texto", "ley", "regulación"
        ]
        
        # Palabras clave que sugieren consultas generales
        general_keywords = [
            "qué es", "cómo funciona", "explica", "define", "significa",
            "cuál es", "dime sobre", "habla de", "información sobre"
        ]
        
        is_document_specific = any(keyword in query_lower for keyword in document_keywords)
        is_general_query = any(keyword in query_lower for keyword in general_keywords)
        
        return {
            "is_document_specific": is_document_specific,
            "is_general_query": is_general_query,
            "query_length": len(query)
        }

    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Ruta optimizada: Redis → RAG → CAG para TODAS las consultas
        Returns:
            dict: {"response": str, "route": str, "source": str, "query_intent": dict}
        """
        try:
            # 0. Analizar intención de la consulta
            query_intent = self._extract_query_intent(query)
            logger.debug(f"Intención de consulta: {query_intent}")
            
            # 1. Revisar si ya existe en Redis
            cached_response = redis_cache.get_cached_response(query)
            if cached_response:
                logger.info("Respuesta obtenida desde Redis")
                return {
                    "response": cached_response, 
                    "route": "redis", 
                    "source": "redis_cache",
                    "query_intent": query_intent
                }

            # 2. Siempre intentar RAG primero (mejorado)
            logger.debug("Ruta optimizada → RAG primero")
            rag_response = rag_engine.generate_answer(query)
            
            # Verificar si RAG tuvo éxito
            if not self._should_fallback_to_cag(rag_response):
                # RAG exitoso → Cachear y retornar
                self._cache_response_if_valid(query, rag_response, "rag")
                return {
                    "response": rag_response, 
                    "route": "rag", 
                    "source": "chroma_db",
                    "query_intent": query_intent
                }
            else:
                # RAG falló → Fallback a CAG con contexto completo optimizado
                logger.warning("RAG no encontró contexto → fallback a CAG con texto completo optimizado")
                
                # Obtener texto completo
                full_text = rag_engine.get_document_text()
                
                if not full_text or not full_text.strip():
                    logger.error("No hay documentos disponibles para CAG")
                    return {
                        "response": "No tengo información suficiente para responder esta pregunta.", 
                        "route": "cag", 
                        "source": "no_documents",
                        "query_intent": query_intent
                    }
                
                logger.debug(f"Longitud del texto completo: {len(full_text)} caracteres")
                
                # Usar CAG con contexto completo optimizado
                cag_response = cag_engine.generate_response(query, external_context=full_text)
                
                # Verificar si CAG encontró la respuesta
                if not self._is_negative_response(cag_response):
                    # CAG exitoso → Cachear
                    self._cache_response_if_valid(query, cag_response, "cag")
                    return {
                        "response": cag_response, 
                        "route": "cag", 
                        "source": "full_context",
                        "query_intent": query_intent
                    }
                else:
                    # Ambos fallaron → Respuesta negativa (no cachear)
                    logger.warning("RAG y CAG fallaron para la consulta")
                    return {
                        "response": "No lo sé", 
                        "route": "fallback", 
                        "source": "no_information",
                        "query_intent": query_intent
                    }
                
        except Exception as e:
            logger.error(f"Error en route_query: {str(e)}")
            # Fallback de emergencia (NO cachear errores)
            emergency_response = "Lo siento, ocurrió un error al procesar tu consulta."
            return {
                "response": emergency_response, 
                "route": "error", 
                "source": "system_error",
                "query_intent": {"error": str(e)}
            }

    def force_rag_search(self, query: str) -> Dict[str, Any]:
        """
        Fuerza una búsqueda RAG ignorando Redis cache
        Útil para testing y debugging
        """
        try:
            logger.debug(f"Forzando búsqueda RAG para: {query}")
            response = rag_engine.generate_answer(query)
            
            return {
                "response": response,
                "route": "rag_forced",
                "source": "chroma_db",
                "cache_ignored": True
            }
            
        except Exception as e:
            logger.error(f"Error en force_rag_search: {str(e)}")
            return {
                "response": "Error en búsqueda forzada",
                "route": "error",
                "source": "system_error"
            }

    def force_cag_search(self, query: str) -> Dict[str, Any]:
        """
        Fuerza una búsqueda CAG con contexto completo
        Útil para testing and debugging
        """
        try:
            logger.debug(f"Forzando búsqueda CAG para: {query}")
            full_text = rag_engine.get_document_text()
            response = cag_engine.generate_response(query, external_context=full_text)
            
            return {
                "response": response,
                "route": "cag_forced",
                "source": "full_context",
                "cache_ignored": True
            }
            
        except Exception as e:
            logger.error(f"Error en force_cag_search: {str(e)}")
            return {
                "response": "Error en búsqueda forzada",
                "route": "error",
                "source": "system_error"
            }

# Instancia singleton para importar
orchestrator = Orchestrator()