import logging
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, vector_store, memory, llm_service, prompt_template: str):
        self.vector_store = vector_store
        self.memory = memory
        self.llm_service = llm_service
        self.prompt_template = PromptTemplate.from_template(prompt_template)

    def run(self, question: str) -> str:
        documents = self.vector_store.retrieve_documents(question=question)
        history = self.memory.get_history()
        context = "\n".join(doc.page_content for doc in documents).strip()

        prompt = self.prompt_template.format(
            context=context,
            question=question,
            history=history
        )

        answer = self.llm_service.invoke(prompt)
        self.memory.save_context(input=question, output=answer)
        return answer