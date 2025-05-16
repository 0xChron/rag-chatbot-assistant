import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import config
from services import VectorStore
from loaders import PDFLoader

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ingest():
    try: 
        logger.info("Starting document ingestion.")
        loader = PDFLoader(filepath=config.FILEPATH)
        embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
        vector_store = VectorStore(embeddings=embeddings,
                                collection_name=config.COLLECTION_NAME,
                                connection_string=config.CONNECTION_STRING)
        
        vector_store.add_documents(documents=loader.load())
        logger.info("Documents ingested succesfully.")
    except Exception as e:
        logger.error(f"Error on document ingestion: {e}")

if __name__ == "__main__":
    ingest()
    
