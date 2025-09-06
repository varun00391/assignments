# db.py
import os
from datetime import datetime
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "rag_chatbot")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

def save_chat(session_id: str, user_query: str, bot_answer: str, source_files=None):
    """Save a chat exchange into MongoDB"""
    record = {
        "session_id": session_id,
        "timestamp": datetime.utcnow(),
        "user_query": user_query,
        "bot_answer": bot_answer,
        "source_files": source_files or []
    }
    db.chats.insert_one(record)

def get_chat_history(session_id: str):
    """Retrieve all chat messages for a session"""
    return list(db.chats.find({"session_id": session_id}).sort("timestamp", 1))
