import os 
from dotenv import load_dotenv

load_dotenv(override=True)

# LLM configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 180000

# Embeddings configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-exp-03-07")

# Database configuration 
CONNECTION_STRING = os.getenv("CONNECTION_STRING", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "test_collection")
