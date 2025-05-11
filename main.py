import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

MODEL_NAME = "gemini-2.0-flash"
TEMPERATURE = 0.2

class GoogleLLM:
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatGoogleGenerativeAI(model=self.model_name, 
                                          temperature=self.temperature)

    def invoke(self, question: str):
        return self.llm.invoke(question)


def main():
    chatbot = GoogleLLM(model_name=MODEL_NAME, temperature=TEMPERATURE)
    response = chatbot.invoke("What is the capital of the Philippines?")
    print(response.content)

if __name__ == "__main__":
    main()