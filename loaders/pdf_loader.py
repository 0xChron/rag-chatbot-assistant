import os
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
    

    def load(self) -> List[Document]:
        try:
            files = os.listdir(self.filepath)
            
            all_docs = []  
            for file in files:
                if file.endswith(".pdf"):
                    docs = PyPDFLoader(os.path.join(self.filepath, file)).load()
                    print(f"Loader {file} with {len(docs)} pages.")
                    all_docs.extend(docs)
            return all_docs
        except Exception as e:
            logger.error(f"Error in PDF load: {e}")
            raise RuntimeError(f"Error in PDF load: {e}")