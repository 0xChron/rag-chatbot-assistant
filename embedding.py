from langchain_google_genai import GoogleGenerativeAIEmbeddings

class Embeddings:
    def __init__(self, model):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model)

    def get_embeddings(self):
        return self.embeddings
