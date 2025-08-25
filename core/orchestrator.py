# core/orchestrator.py
import logging
import re
from config.settings import RAG_THRESHOLD

logger = logging.getLogger(__name__)

class Orchestrator:
    """Orquestador inteligente que decide entre RAG y CAG para documentos legales"""
    
    def __init__(self):
        self.rag_threshold = RAG_THRESHOLD
        logger.info(f"✅ Orquestador inicializado con umbral RAG: {self.rag_threshold}")
    
    def should_use_rag(self, query):
        """
        Decide si usar RAG basándose en el tipo de consulta legal/laboral
        """
        query_lower = query.lower().strip()
        
        # 1. Consultas que SIEMPRE usan CAG (conversacionales/generales)
        if self._is_general_conversation(query_lower):
            logger.debug("💬 Consulta conversacional → usar CAG")
            return False
        
        # 2. Consultas que SIEMPRE usan RAG (específicas de documentos)
        if self._is_document_specific_query(query_lower):
            logger.debug("📄 Consulta específica de documento → usar RAG")
            return True
        
        # 3. Consultas basadas en patrones legales/laborales
        if self._is_legal_labor_query(query_lower):
            logger.debug("⚖️ Consulta legal/laboral → usar RAG")
            return True
        
        # 4. Por defecto para consultas ambiguas
        logger.debug("⚡ Consulta ambigua → usar RAG por defecto")
        return True

    def _is_general_conversation(self, query_lower):
        """Determina si es una conversación general que no requiere documentos"""
        general_patterns = [
            r"^hola.*", r"^buenos días.*", r"^buenas tardes.*", r"^buenas noches.*",
            r"^cómo estás.*", r"^quién eres.*", r"^qué puedes hacer.*", 
            r"^gracias.*", r"^adiós.*", r"^chao.*",
            r"^qué hora es.*", r"^dime un chiste.*", r"^habla sobre ti.*",
            r"^ok.*", r"^entendido.*", r"^perfecto.*", r"^excelente.*",
            r"^genial.*", r"^bien.*", r"^vale.*", r"^de acuerdo.*"
        ]
        
        return any(re.match(pattern, query_lower) for pattern in general_patterns)

    def _is_document_specific_query(self, query_lower):
        """Determina si la consulta hace referencia específica a documentos subidos"""
        document_patterns = [
            r".*documento.*", r".*pdf.*", r".*archivo.*", r".*contrato.*",
            r".*ley.*", r".*norma.*", r".*reglamento.*", r".*cláusula.*",
            r".*artículo.*", r".*inciso.*", r".*sección.*", r".*capítulo.*",
            r"en el (pdf|documento).*", r"según (el|la).*", r"de acuerdo a.*",
            r"basado en.*", r"según lo establecido.*", r"conforme a.*"
        ]
        
        return any(re.search(pattern, query_lower) for pattern in document_patterns)

    def _is_legal_labor_query(self, query_lower):
        """Determina si es una consulta legal/laboral que requiere documentos"""
        legal_patterns = [
            # Patrones laborales
            r"jornada laboral.*", r"horario de trabajo.*", r"salario.*", r"sueldo.*",
            r"vacaciones.*", r"días libres.*", r"descanso.*", r"horas extras.*",
            r"bonificaciones.*", r"prestaciones.*", r"aguinaldo.*", r"utilidades.*",
            r"prima vacacional.*", r"fondo de ahorro.*", r"vales de despensa.*",
            
            # Patrones legales
            r"está prohibido.*", r"está permitido.*", r"obligaciones.*",
            r"derechos.*", r"deberes.*", r"prohibición.*", r"permiso.*",
            r"multa.*", r"sanción.*", r"infracción.*", r"incumplimiento.*",
            r"responsabilidad.*", r"cláusula.*", r"contrato.*", r"convenio.*",
            
            # Consultas específicas
            r"qué pasa si.*", r"consecuencias de.*", r"cómo solicitar.*",
            r"requisitos para.*", r"procedimiento.*", r"trámite.*",
            r"cómo funciona.*", r"qué necesito para.*", r"pasos para.*",
            
            # Palabras clave legales
            r"contratación.*", r"despido.*", r"renuncia.*", r"liquidación.*",
            r"jubilación.*", r"pensión.*", r"seguro social.*", r"imss.*",
            r"infonavit.*", r"fonacot.*", r"capacitación.*", r"inducción.*"
        ]
        
        return any(re.search(pattern, query_lower) for pattern in legal_patterns)
    
    def route_query(self, query):

        query_lower = query.lower().strip()
        
        # Verificar si es conversación general
        is_general = self._is_general_conversation(query_lower)
        
        if is_general:
            logger.debug("💬 Consulta conversacional → usar CAG (modo general)")
            return "cag", True
        
        # Decidir entre RAG y CAG documental
        if self.should_use_rag(query):
            return "rag", False
        else:
            return "cag", False

# Instancia singleton para ser importada
orchestrator = Orchestrator()