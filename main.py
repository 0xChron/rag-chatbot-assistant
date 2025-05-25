import os
import uuid
import logging
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import config
from services import PromptBuilder, VectorStore, LLMService, MemoryService

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def app():
    logger.info("Starting chatbot app.")
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
    chatbot = LLMService(model_name=config.LLM_MODEL, 
        temperature=config.LLM_TEMPERATURE)
    vector_store = VectorStore(embeddings=embeddings, 
        collection_name=config.COLLECTION_NAME, 
        connection_string=config.CONNECTION_STRING)
    memory = MemoryService(
        connection_string=config.MEMORY_CONNECTION_STRING,
        session_id=str(uuid.uuid4()), 
        table_name=config.TABLE_NAME,
        k=3
    )
    
    # memory.create_table() # Create the table if it doesn't exist

    while True:
        question = input("User (type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        prompt = PromptBuilder(
            question=question,
            documents=vector_store.retrieve_documents(question=question),
            history=memory.get_history()
        ).format_prompt()

        answer = chatbot.invoke(prompt)
        logger.info(f"AI: {answer}")
        
        memory.save_context(input=question, output=answer)

if __name__ == "__main__":
    app()