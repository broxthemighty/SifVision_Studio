import sqlite3
import os

class MemoryManager:
    def __init__(self, path="./data/memory.db"):
        self.conn = sqlite3.connect(path)
        self.create_table()

    def create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_text TEXT,
            response_text TEXT
        )
        """)

    def store_entry(self, user_text, response_text):
        self.conn.execute("INSERT INTO memory (user_text, response_text) VALUES (?, ?)", (user_text, response_text))
        self.conn.commit()
