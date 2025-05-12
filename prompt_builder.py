from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing import List

class PromptBuilder:
    def __init__(self, question: str, documents: List[Document]):
        self.question = question
        self.documents = documents


    def format_prompt(self) -> str:
        document_content = [doc.page_content for doc in self.documents]
        context = "\n".join(document_content).strip()

        prompt = PromptTemplate.from_template(
            """You are an AI assistant helping with questions based on provided context.
            
            Context:
            {context}
            
            Question:
            {question}
            
            Helpful Answer:
            """
        )
        return prompt.format(context=context, question=self.question)