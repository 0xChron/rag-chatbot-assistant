import logging
from langchain_google_genai import ChatGoogleGenerativeAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._initialize_google_llm()

    def _initialize_google_llm(self) -> ChatGoogleGenerativeAI:
        try:
            return ChatGoogleGenerativeAI(model=self.model_name,
                                        temperature=self.temperature)
        except Exception as e:
            logger.error(f"Error initializing Google LLM: {e}")
            raise RuntimeError(f"Error initializing Google LLM: {e}")
    
    def invoke(self, question: str):
        try:
            return self.llm.invoke(question)
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            raise RuntimeError(f"Error invoking LLM: {e}")