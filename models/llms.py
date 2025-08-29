# models/llms.py
import os
import time
import logging
from openai import AzureOpenAI
from config.settings import AZURE_OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_API_VERSION, OPENAI_MODEL_NAME

logger = logging.getLogger(__name__)

class AzureOpenAIClient:
    """Cliente optimizado para interactuar con Azure OpenAI con reintentos"""
    
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=OPENAI_API_KEY,
            api_version=OPENAI_API_VERSION,
        )
        self.model_name = OPENAI_MODEL_NAME
        logger.info(f"Cliente de Azure OpenAI configurado con modelo: {self.model_name}")
    
    def generate_response(self, messages, temperature=0.1, max_tokens=2000, max_retries=3):
        """Genera una respuesta con reintentos automáticos para fallos transitorios"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
                
            except Exception as e:
                # Si es el último intento, relanza la excepción
                if attempt == max_retries - 1:
                    logger.error(f"Error en la generación con Azure OpenAI después de {max_retries} intentos: {str(e)}")
                    raise
                
                # Esperar antes de reintentar (backoff exponencial)
                wait_time = 2 ** attempt
                logger.warning(f"Intento {attempt + 1} fallado. Reintentando en {wait_time} segundos. Error: {str(e)}")
                time.sleep(wait_time)
                continue

    def test_connection(self):
        """Prueba la conexión con Azure OpenAI"""
        try:
            test_messages = [{"role": "user", "content": "Responde 'OK' para prueba de conexión"}]
            response = self.generate_response(test_messages, max_tokens=10, max_retries=1)
            return f"Conexión exitosa: {response}"
        except Exception as e:
            return f"Error de conexión: {str(e)}"

# Instancia singleton para ser importada
azure_llm_client = AzureOpenAIClient()