# config.py
import os
from dataclasses import dataclass


@dataclass
class Config:

    # -------------------------
    # Groq (LLM for QA)
    # -------------------------
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_EXPAND_MODEL: str = os.getenv("GROQ_EXPAND_MODEL", "llama-3.1-8b-instant")

    # -------------------------
    # Embeddings (local model)
    # -------------------------
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------------------------
    # Vector Database
    # -------------------------
    VECTOR_DB_TYPE: str = "chromadb"
    CHROMA_DB_PATH: str = "./data/chroma_db"

    # -------------------------
    # Document Processing
    # -------------------------
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_DOCUMENT_SIZE: int = 50 * 1024 * 1024  # 50MB

    # -------------------------
    # Memory Settings
    # -------------------------
    SHORT_TERM_MEMORY_SIZE: int = 20
    SESSION_TIMEOUT: int = 3600  # seconds

    # -------------------------
    # Learning / Feedback
    # -------------------------
    FEEDBACK_STORAGE_PATH: str = "./data/feedback"
    MODEL_CACHE_PATH: str = "./data/models"

    # -------------------------
    # Performance
    # -------------------------
    CACHE_TTL: int = 3600
    MAX_CONCURRENT_REQUESTS: int = 10


config = Config()