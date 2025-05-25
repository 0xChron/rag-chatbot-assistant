import uuid
import logging
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import config
from services import VectorStore, LLMService, MemoryService
from pipelines import RAGPipeline

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def app():
    logger.info("Starting chatbot app.")
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
    vector_store = VectorStore(embeddings=embeddings, 
        collection_name=config.COLLECTION_NAME, 
        connection_string=config.CONNECTION_STRING
    )
    memory = MemoryService(
        connection_string=config.MEMORY_CONNECTION_STRING,
        session_id=str(uuid.uuid4()), 
        table_name=config.TABLE_NAME,
        k=3
    )
    llm_service = LLMService(model_name=config.LLM_MODEL, 
        temperature=config.LLM_TEMPERATURE
    )

    rag = RAGPipeline(vector_store, memory, llm_service, config.PROMPT_TEMPLATE)

    while True:
        question = input("User (type 'exit' to quit): ")
        if question.lower() == "exit":
            break

        answer = rag.run(question)
        print(f"AI: {answer}")

if __name__ == "__main__":
    app()