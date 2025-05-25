import os 
from dotenv import load_dotenv

load_dotenv(override=True)

# Data source configuration
FILEPATH = os.getenv("FILEPATH", "data")

# LLM configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 180000

# Embeddings configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-exp-03-07")

# Database configuration 
CONNECTION_STRING = os.getenv("CONNECTION_STRING", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "test_collection")

# Memory configuration
HOST = os.getenv("HOST", "")
DBNAME = os.getenv("DBNAME", "")
USER = os.getenv("USER", "")
PASSWORD = os.getenv("PASSWORD", "")
PORT = os.getenv("PORT", "")
MEMORY_CONNECTION_STRING = f"host={HOST} dbname={DBNAME} user={USER} password={PASSWORD} port={PORT}"
TABLE_NAME = os.getenv("TABLE_NAME", "chat_history")

# Prompt configuration
PROMPT_TEMPLATE = """
You are an AI assistant helping with questions based on provided context.

Context: 
{context}

Question:
{question}

Chat History:
{history}

Helpful Answer:
"""
