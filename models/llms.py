import os
os.environ['NO_PROXY'] = 'recursoazureopenaimupi.openai.azure.com'
import logging
import time
from typing import List, Dict, Optional
from openai import AzureOpenAI, APIError, APIConnectionError, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config.settings import AZURE_OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_API_VERSION, OPENAI_MODEL_NAME

logger = logging.getLogger(__name__)

# Excepciones que deben triggerear retry
RETRY_EXCEPTIONS = (APIError, APIConnectionError, APITimeoutError, RateLimitError)

class AzureOpenAIClient:
    """Cliente robusto para interactuar con Azure OpenAI con manejo avanzado de errores"""
    
    def __init__(self):
        try:
            self.client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=OPENAI_API_KEY,
                api_version=OPENAI_API_VERSION,
                timeout=30.0,  
                max_retries=3  
            )
            self.model_name = OPENAI_MODEL_NAME
            logger.info(f"✅ Cliente de Azure OpenAI configurado con modelo: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Error crítico inicializando Azure OpenAI: {str(e)}")
            raise
    
    @retry(
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=lambda retry_state: logger.warning(
            f"⚠️ Reintentando llamada a Azure OpenAI (intento {retry_state.attempt_number}/3): {retry_state.outcome.exception()}"
        )
    )
    def generate_response(self, messages: List[Dict], temperature: float = 0.1, max_tokens: int = 1500) -> str:
        """
        Genera una respuesta usando el chat completion de Azure OpenAI con retry automático
        """
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            processing_time = time.time() - start_time
            response_content = response.choices[0].message.content
            
            logger.info(f"✅ Respuesta generada en {processing_time:.2f}s - Tokens: {response.usage.total_tokens}")
            return response_content
            
        except RETRY_EXCEPTIONS as e:
            logger.warning(f"⚠️ Error de API (será reintentado): {type(e).__name__}: {str(e)}")
            raise  
            
        except Exception as e:
            logger.error(f"❌ Error no recuperable en Azure OpenAI: {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"Error no recuperable al generar respuesta: {str(e)}")
    
    def generate_legal_response(self, messages: List[Dict]) -> str:
        """
        Configuración optimizada para respuestas legales/laborales
        """
        legal_params = {
            "temperature": 0.1,     
            "max_tokens": 2000,      
            "top_p": 0.9,
            "frequency_penalty": 0.2, 
            "presence_penalty": 0.1
        }
        
        return self.generate_response(messages, **legal_params)
    
    def generate_conversational_response(self, messages: List[Dict]) -> str:
        """
        Configuración optimizada para conversación general
        """
        conversational_params = {
            "temperature": 0.3,      
            "max_tokens": 800,      
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        return self.generate_response(messages, **conversational_params)
    
    def health_check(self) -> bool:
        """Verifica que el cliente esté funcionando correctamente"""
        try:
            # Llamada de health check
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hola"}],
                max_tokens=5
            )
            return test_response.choices[0].message.content is not None
        except Exception as e:
            logger.error(f"❌ Health check falló: {str(e)}")
            return False

# Instancia singleton para ser importada
try:
    azure_llm_client = AzureOpenAIClient()
    
    # Health check inicial
    if azure_llm_client.health_check():
        logger.info("✅ Health check de Azure OpenAI exitoso")
    else:
        logger.warning("⚠️ Health check de Azure OpenAI arrojó resultados inesperados")
        
except Exception as e:
    logger.critical(f"💥 Error crítico inicializando Azure OpenAI Client: {str(e)}")
    # Fallback a una instancia dummy para evitar crash total
    class FallbackClient:
        def generate_response(self, *args, **kwargs):
            return "Lo siento, el servicio de IA no está disponible en este momento."
        def generate_legal_response(self, *args, **kwargs):
            return "Lo siento, el servicio de IA no está disponible para consultas legales."
        def generate_conversational_response(self, *args, **kwargs):
            return "Lo siento, no puedo responder en este momento. Por favor, intenta más tarde."
        def health_check(self):
            return False
    
    azure_llm_client = FallbackClient()
    logger.error("🔄 Usando cliente de fallback por error de inicialización")