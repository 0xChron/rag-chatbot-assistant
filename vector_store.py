from typing import List, Any
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from pdf_loader import PDFLoader

class VectorStore:
    def __init__(self, embeddings: Any, collection_name: str, connection_string: str):
        self.store = PGVector(
            collection_name=collection_name,
            connection=connection_string,
            embeddings=embeddings
        )


    def add_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) 
        
        split_docs = []
        for doc in documents:
            chunks = splitter.split_text(doc.page_content)
            split_docs.extend([Document(page_content=chunk, metadata=doc.metadata) for chunk in chunks])
        self.store.add_documents(split_docs)


    def retrieve_documents(self, question: str, k: int = 5) -> List[Document]:
        retriever = self.store.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(question)