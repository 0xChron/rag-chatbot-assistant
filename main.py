import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Testing modules
import config
from prompt_builder import PromptBuilder
from vector_store import VectorStore

load_dotenv()


class GoogleLLM:
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatGoogleGenerativeAI(model=self.model_name, 
                                          temperature=self.temperature)

    def invoke(self, question: str):
        return self.llm.invoke(question)


def main():
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
    chatbot = GoogleLLM(model_name=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)

    vector_store = VectorStore(embeddings=embeddings, 
                            collection_name=config.COLLECTION_NAME, 
                            connection_string=config.CONNECTION_STRING)
    
    question = "What is the course code of the subjects?"

    prompt = PromptBuilder(question=question, 
                           documents=vector_store.retrieve_documents(question=question)).format_prompt()
    print(prompt)
    response = chatbot.invoke(prompt)
    print(response.content)

if __name__ == "__main__":
    main()