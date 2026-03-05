# src/memory_system.py
from __future__ import annotations

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import logging

from .embedding_service import EmbeddingService
from .vector_db import VectorDatabase

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    content: Any
    timestamp: float
    memory_type: str
    metadata: Dict[str, Any]


class MemorySystem:
    def __init__(
        self,
        vector_db: VectorDatabase,
        embedding_service: EmbeddingService,
        short_term_size: int = 20,
    ):
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.short_term_size = short_term_size

        # session_id -> list[MemoryItem]
        self.sessions: Dict[str, List[MemoryItem]] = {}

    # -------------------------
    # Short-term memory (per session)
    # -------------------------
    def add_to_short_term_memory(
        self,
        session_id: str,
        query: str,
        answer: str,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> None:
        item = MemoryItem(
            content={"query": query, "answer": answer, "feedback": feedback or {}},
            timestamp=time.time(),
            memory_type="short_term",
            metadata={"session_id": session_id},
        )

        self.sessions.setdefault(session_id, []).append(item)

        # keep only last N
        if len(self.sessions[session_id]) > self.short_term_size:
            self.sessions[session_id] = self.sessions[session_id][-self.short_term_size :]

    def get_short_term_context(self, session_id: str) -> List[Dict[str, Any]]:
        if session_id not in self.sessions:
            return []
        return [item.content for item in self.sessions[session_id][-self.short_term_size :]]

    # -------------------------
    # Long-term memory (semantic)
    # -------------------------
    def add_to_long_term_memory(self, question: str, answer: str, topic: str, confidence: float) -> None:
        """
        Store successful Q&A to vector DB with embeddings for semantic retrieval.
        """
        try:
            text = f"Q: {question}\nA: {answer}"
            embedding = self.embedding_service.generate_embeddings([text])[0]
            self.vector_db.store_qa_pair(question, answer, topic, confidence, embedding=embedding)
            logger.info(f"Added Q&A to long-term memory: topic={topic}")
        except Exception:
            logger.exception("Error adding to long-term memory")

    def search_long_term_memory(self, query: str, topic: Optional[str] = None, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Semantic search over saved Q&A pairs.
        """
        try:
            q_emb = self.embedding_service.generate_embeddings([query])[0]
            return self.vector_db.search_qa_pairs(query_embedding=q_emb, topic=topic, limit=limit)
        except Exception:
            logger.exception("Error searching long-term memory")
            return []

    # -------------------------
    # Episodic (history)
    # -------------------------
    def get_episodic_memory(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session from vector database.
        """
        try:
            return self.vector_db.get_conversation_history(session_id)
        except Exception as e:
            logger.exception(f"Error getting episodic memory for session {session_id}")
            return []

    def cleanup_old_sessions(self, max_age_seconds: int = 3600) -> None:
        now = time.time()
        to_remove = []
        for session_id, items in self.sessions.items():
            if items and (now - items[0].timestamp) > max_age_seconds:
                to_remove.append(session_id)

        for session_id in to_remove:
            del self.sessions[session_id]
            logger.info(f"Cleaned up old session: {session_id}")