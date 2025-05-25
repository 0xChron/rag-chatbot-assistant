import logging
from typing import Any
from langchain.memory import ConversationBufferWindowMemory
from langchain_postgres import PostgresChatMessageHistory

import config
import psycopg

class MemoryService:
    def __init__(self, connection_string: str, session_id: str, table_name: str, k: int = 5):
        self.connection = psycopg.connect(connection_string)
        self.session_id = session_id
        self.table_name = table_name
        self.k = k
        self.chat_history = PostgresChatMessageHistory(
            self.table_name,
            self.session_id,
            sync_connection=self.connection
        )
        self.memory = self._initialize_memory()

    def __repr__(self) -> str:
        return f"<MemoryService(session_id={self.session_id}, table_name={self.table_name})>"

    def _initialize_memory(self) -> ConversationBufferWindowMemory:
        return ConversationBufferWindowMemory(chat_memory=self.chat_history, 
                                              k=self.k, 
                                              return_messages=True)

    def save_context(self, input: str, output: str) -> None:
        return self.memory.save_context({"input": input}, {"output": output})
    
    def get_history(self) -> str:
        history_messages = self.memory.load_memory_variables({})["history"]
        return "\n".join([f"User: {message.content}" if message.type == "human" else f"AI: {message.content}" for message in history_messages])
    
    def create_table(self) -> None:
        self.chat_history.create_tables(self.connection, self.table_name)


