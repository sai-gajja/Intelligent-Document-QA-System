# src/vector_db.py
from __future__ import annotations

import time
import uuid
import logging
from typing import List, Dict, Any, Optional

import chromadb
from config import config

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.collections = self._initialize_collections()

    def _initialize_collections(self):
        collections = {}
        try:
            collections["document_chunks"] = self.client.get_or_create_collection(
                name="document_chunks",
                metadata={"description": "Document chunks with embeddings"},
            )
            collections["user_interactions"] = self.client.get_or_create_collection(
                name="user_interactions",
                metadata={"description": "User query and feedback history"},
            )
            collections["feedback_data"] = self.client.get_or_create_collection(
                name="feedback_data",
                metadata={"description": "Explicit and implicit feedback"},
            )
            collections["qa_pairs"] = self.client.get_or_create_collection(
                name="qa_pairs",
                metadata={"description": "Successful Q&A pairs"},
            )
            logger.info("All collections initialized successfully")
            return collections
        except Exception as e:
            logger.exception("Error initializing collections")
            raise e

    # -------------------------
    # Document chunks
    # -------------------------
    def upsert_document_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have same length")

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [self._sanitize_metadata(c["metadata"]) for c in chunks]

        col = self.collections["document_chunks"]

        try:
            try:
                col.delete(ids=ids)
            except Exception:
                pass

            col.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            logger.info(f"Upserted {len(chunks)} document chunks")
        except Exception as e:
            logger.exception("Error storing document chunks")
            raise e

    def search_similar_chunks(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        dedupe_by: str = "id",
    ) -> List[Dict[str, Any]]:
        try:
            # Handle multiple document IDs
            where_filter = None
            if filters and "doc_id" in filters:
                doc_ids = filters["doc_id"]
                if isinstance(doc_ids, list):
                    # If it's a list of doc_ids, create an OR condition
                    if len(doc_ids) == 1:
                        where_filter = {"doc_id": doc_ids[0]}
                    elif len(doc_ids) > 1:
                        # For multiple doc_ids, use $or operator
                        where_filter = {
                            "$or": [{"doc_id": doc_id} for doc_id in doc_ids]
                        }
                else:
                    # Single doc_id as string
                    where_filter = {"doc_id": doc_ids}
            
            logger.info(f"Searching with filter: {where_filter}")
            
            results = self.collections["document_chunks"].query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,  # Get more results to account for filtering
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            docs = (results.get("documents") or [[]])[0]
            metas = (results.get("metadatas") or [[]])[0]
            dists = (results.get("distances") or [[]])[0]
            ids = (results.get("ids") or [[]])[0]

            out: List[Dict[str, Any]] = []
            seen = set()

            for i in range(len(docs)):
                doc = docs[i]
                meta = metas[i] if i < len(metas) else {}
                dist = float(dists[i]) if i < len(dists) and dists[i] is not None else 0.0
                _id = ids[i] if i < len(ids) else None

                key = _id if dedupe_by == "id" else (meta.get(dedupe_by), _id)
                if key in seen:
                    continue
                seen.add(key)

                out.append(
                    {
                        "content": doc,
                        "metadata": meta,
                        "distance": dist,
                        "score": 1.0 / (1.0 + dist),
                        "id": _id,
                    }
                )

            # Sort by score and limit to n_results
            out.sort(key=lambda x: x["score"], reverse=True)
            return out[:n_results]
            
        except Exception as e:
            logger.exception("Error searching similar chunks")
            return []

    # -------------------------
    # Interactions + Feedback
    # -------------------------
    def store_user_interaction(
        self,
        session_id: str,
        query: str,
        answer: str,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> str:
        try:
            interaction_id = str(uuid.uuid4())
            ts = time.time()
            
            logger.info(f"Generating interaction_id: {interaction_id} for session: {session_id}")

            metadata = {
                "session_id": session_id,
                "query": query,
                "answer": answer,
                "timestamp": ts,
                "has_feedback": bool(feedback),
                "interaction_id": interaction_id,
            }

            doc_text = f"Q: {query}\nA: {answer}"
            if feedback:
                doc_text += f"\nFEEDBACK: {str(feedback)}"

            self.collections["user_interactions"].add(
                ids=[interaction_id],
                documents=[doc_text],
                metadatas=[metadata],
            )
            
            logger.info(f"Successfully stored interaction {interaction_id} for session {session_id}")
            return interaction_id
            
        except Exception as e:
            logger.exception(f"Error storing user interaction: {e}")
            return str(uuid.uuid4())

    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Getting conversation history for session {session_id}")
            
            results = self.collections["user_interactions"].get(
                where={"session_id": session_id},
                limit=limit,
                include=["metadatas", "documents"]
            )
            
            ids = results.get("ids", [])
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            
            conversations = []
            for i in range(len(ids)):
                metadata = metadatas[i] if i < len(metadatas) else {}
                
                conversation = {
                    "id": ids[i] if i < len(ids) else None,
                    "query": metadata.get("query", ""),
                    "answer": metadata.get("answer", ""),
                    "timestamp": metadata.get("timestamp", 0),
                    "has_feedback": metadata.get("has_feedback", False),
                    "content": documents[i] if i < len(documents) else ""
                }
                conversations.append(conversation)
            
            conversations.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            return conversations
            
        except Exception as e:
            logger.exception(f"Error getting conversation history: {e}")
            return []

    def store_feedback(
        self,
        interaction_id: str,
        feedback_type: str,
        feedback_data: Dict[str, Any],
        corrected_answer: Optional[str] = None,
    ) -> bool:
        try:
            feedback_id = str(uuid.uuid4())
            ts = time.time()

            metadata = {
                "interaction_id": interaction_id,
                "feedback_type": feedback_type,
                "timestamp": ts,
                "has_correction": bool(corrected_answer),
                "feedback_data": str(feedback_data)
            }

            doc_text = f"{feedback_type}: {str(feedback_data)}"
            if corrected_answer:
                doc_text += f"\nCORRECTED: {corrected_answer}"

            self.collections["feedback_data"].add(
                ids=[feedback_id],
                documents=[doc_text],
                metadatas=[metadata],
            )
            
            # Also update the original interaction to mark it has feedback
            try:
                self.collections["user_interactions"].update(
                    ids=[interaction_id],
                    metadatas=[{"has_feedback": True}]
                )
            except:
                pass
                
            logger.info(f"Stored feedback {feedback_id} for interaction {interaction_id}")
            return True
            
        except Exception as e:
            logger.exception(f"Error storing feedback: {e}")
            return False

    def store_qa_pair(self, question: str, answer: str, topic: str, confidence: float, embedding: Optional[List[float]] = None) -> None:
        try:
            qa_id = str(uuid.uuid4())
            ts = time.time()

            metadata = {
                "question": question,
                "answer": answer,
                "topic": topic,
                "confidence": float(confidence),
                "usage_count": 1,
                "timestamp": ts,
            }

            doc_text = f"Q: {question}\nA: {answer}"

            if embedding is not None:
                self.collections["qa_pairs"].add(
                    ids=[qa_id],
                    documents=[doc_text],
                    metadatas=[metadata],
                    embeddings=[embedding],
                )
            else:
                self.collections["qa_pairs"].add(
                    ids=[qa_id],
                    documents=[doc_text],
                    metadatas=[metadata],
                )
        except Exception as e:
            logger.exception(f"Error storing Q&A pair: {e}")

    def search_qa_pairs(self, query_embedding: List[float], topic: Optional[str] = None, limit: int = 3) -> List[Dict[str, Any]]:
        try:
            where = {"topic": topic} if topic else None

            results = self.collections["qa_pairs"].query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where,
                include=["metadatas", "documents", "distances"],
            )

            docs = (results.get("documents") or [[]])[0]
            metas = (results.get("metadatas") or [[]])[0]
            dists = (results.get("distances") or [[]])[0]
            ids = (results.get("ids") or [[]])[0]

            out = []
            for i in range(len(docs)):
                meta = metas[i] if i < len(metas) else {}
                dist = float(dists[i]) if i < len(dists) and dists[i] is not None else 0.0
                out.append(
                    {
                        "id": ids[i] if i < len(ids) else None,
                        "question": meta.get("question"),
                        "answer": meta.get("answer"),
                        "topic": meta.get("topic"),
                        "confidence": meta.get("confidence"),
                        "usage_count": meta.get("usage_count"),
                        "distance": dist,
                        "score": 1.0 / (1.0 + dist),
                    }
                )

            return out
        except Exception as e:
            logger.exception(f"Error searching Q&A pairs: {e}")
            return []

    def _sanitize_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        safe: Dict[str, Any] = {}
        for k, v in (meta or {}).items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                safe[k] = v
            else:
                safe[k] = str(v)
        return safe