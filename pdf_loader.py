import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

class PDFLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
    

    def load(self) -> List[Document]:
        files = os.listdir(self.filepath)
        
        all_docs = []  
        for file in files:
            if file.endswith(".pdf"):
                docs = PyPDFLoader(os.path.join(self.filepath, file)).load()
                print(f"Loader {file} with {len(docs)} pages.")
                all_docs.extend(docs)
        return all_docs