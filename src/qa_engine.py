# src/qa_engine.py
from __future__ import annotations

import os
import time
import hashlib
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

from groq import Groq

from .embedding_service import EmbeddingService
from .vector_db import VectorDatabase
from .memory_system import MemorySystem

logger = logging.getLogger(__name__)

class QAEngine:
    """
    Groq-powered RAG QA Engine with multi-document support.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_db: VectorDatabase,
        memory_system: MemorySystem,
        groq_api_key: Optional[str] = None,
        answer_model: Optional[str] = None,
        expand_model: Optional[str] = None,
        enable_query_expansion: bool = True,
        cache_ttl_seconds: int = 300,
        max_context_chars: int = 12000,
    ):
        self.embedding_service = embedding_service
        self.vector_db = vector_db
        self.memory_system = memory_system

        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Missing GROQ_API_KEY. Set it in environment or pass groq_api_key.")

        self.client = Groq(api_key=self.groq_api_key)

        self.answer_model = answer_model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.expand_model = expand_model or os.getenv("GROQ_EXPAND_MODEL", "llama-3.1-8b-instant")
        self.enable_query_expansion = enable_query_expansion

        self.qa_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_context_chars = max_context_chars

    def process_query(
        self,
        query: str,
        session_id: str,
        document_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()

        try:
            # Log the document filters
            if document_filters and "doc_id" in document_filters:
                doc_ids = document_filters["doc_id"]
                if isinstance(doc_ids, list):
                    logger.info(f"Searching across {len(doc_ids)} selected documents")
                else:
                    logger.info(f"Searching in single document: {doc_ids[:8] if doc_ids else 'unknown'}")
            else:
                logger.info("Searching across ALL documents")

            # 0) Cache check
            cache_key = self._generate_cache_key(query, session_id, document_filters)
            cached = self._cache_get(cache_key)
            if cached is not None:
                logger.info("Cache hit for query")
                return cached

            # 1) Query expansion
            expanded_query = query
            if self.enable_query_expansion and self._needs_expansion(query):
                expanded_query = self._expand_query(query, session_id)

            # 2) Embed query
            query_embedding = self.embedding_service.generate_embeddings([expanded_query])[0]

            # 3) Retrieve chunks - with better handling for multiple docs
            relevant_chunks = self.vector_db.search_similar_chunks(
                query_embedding,
                n_results=10,  # Increased for better coverage across multiple docs
                filters=document_filters,
            )

            # Log which documents we found chunks from
            if relevant_chunks:
                doc_ids = set()
                for chunk in relevant_chunks:
                    doc_id = chunk.get("metadata", {}).get("doc_id")
                    if doc_id:
                        doc_ids.add(doc_id[:8])
                logger.info(f"Found chunks from documents: {list(doc_ids)}")
            else:
                logger.warning("No relevant chunks found")

            # 4) Search long-term memory
            similar_qa = self.memory_system.search_long_term_memory(expanded_query)

            # 5) Prepare context with document names
            context = self._prepare_context(relevant_chunks, similar_qa, session_id)

            # 6) Generate answer with document awareness
            answer = self._generate_answer(query, context, document_filters)

            # 7) Store interaction
            interaction_id = self.vector_db.store_user_interaction(session_id, query, answer)
            logger.info(f"Created interaction_id: {interaction_id} for session: {session_id}")

            # 8) Update short-term memory
            self.memory_system.add_to_short_term_memory(
                session_id,
                query,
                answer,
                feedback={"interaction_id": interaction_id},
            )

            # 9) Calculate confidence
            confidence = self._calculate_confidence(answer, relevant_chunks)

            result = {
                "answer": answer,
                "confidence": confidence,
                "sources": self._format_sources(relevant_chunks),
                "similar_qa": similar_qa,
                "processing_time": time.time() - start_time,
                "interaction_id": interaction_id,
                "expanded_query": expanded_query if expanded_query != query else None,
            }

            # 10) Cache
            self._cache_set(cache_key, result)

            # 11) Long-term memory if high confidence
            if confidence >= 0.80:
                topic = self._extract_topic(query)
                self.memory_system.add_to_long_term_memory(query, answer, topic, confidence)

            return result

        except Exception as e:
            logger.exception("Error processing query")
            return {
                "answer": "I ran into an error while processing your query. Please try again.",
                "confidence": 0.0,
                "sources": [],
                "similar_qa": [],
                "processing_time": time.time() - start_time,
                "interaction_id": None,
                "error": str(e),
            }

    def _generate_answer(self, query: str, context: str, document_filters: Optional[Dict] = None) -> str:
        """Generate answer with document context awareness"""
        
        # Add instruction about which documents to use
        doc_instruction = ""
        if document_filters and "doc_id" in document_filters:
            doc_ids = document_filters["doc_id"]
            if isinstance(doc_ids, list):
                if len(doc_ids) == 1:
                    doc_instruction = "\nImportant: Only use information from the selected document. Do not use information from other documents.\n"
                else:
                    doc_instruction = f"\nImportant: Only use information from the {len(doc_ids)} selected documents. Do not use information from other documents.\n"
        
        system = (
            "You are a document Q&A assistant.\n"
            "Strict rules:\n"
            "- Use ONLY the provided DOCUMENT EXCERPTS for facts.\n"
            "- If the answer is not explicitly supported, say: \"I can't find that in the provided documents.\"\n"
            "- Every factual sentence must end with a citation like [p3].\n"
            "- Do not cite pages you didn't use.\n"
            "- Keep it concise.\n"
            f"{doc_instruction}"
        )

        user = (
            f"Context:\n{self._trim_context(context)}\n\n"
            f"Question: {query}\n\n"
            "Answer (with citations like [p2]):"
        )

        try:
            return self._groq_chat(
                model=self.answer_model,
                user_text=user,
                system_text=system,
                max_tokens=700,
                temperature=0.2,
            )
        except Exception as e:
            logger.warning(f"Primary model failed ({self.answer_model}): {e}")
            fallback = "llama-3.3-70b-versatile" if self.answer_model != "llama-3.3-70b-versatile" else "llama-3.1-8b-instant"
            return self._groq_chat(
                model=fallback,
                user_text=user,
                system_text=system,
                max_tokens=700,
                temperature=0.2,
            )

    # -------------------------
    # Query expansion decision
    # -------------------------
    def _needs_expansion(self, query: str) -> bool:
        q = (query or "").strip().lower()
        if len(q.split()) <= 5:
            return True
        ambiguous = {"this", "that", "it", "they", "he", "she", "those", "these", "above", "earlier", "previous"}
        tokens = set(q.split())
        return len(tokens.intersection(ambiguous)) > 0

    def _expand_query(self, query: str, session_id: str) -> str:
        context = self.memory_system.get_short_term_context(session_id)
        if not context:
            return query

        last_turns = context[-3:]
        context_text = "\n".join([f"Q: {t['query']}\nA: {t['answer']}" for t in last_turns])

        prompt = (
            "Rewrite the user's query to be self-contained and specific using the conversation context.\n"
            "Return ONLY the rewritten query.\n\n"
            f"Conversation:\n{context_text}\n\n"
            f"User query: {query}\n"
            "Rewritten query:"
        )

        try:
            rewritten = self._groq_chat(
                model=self.expand_model,
                user_text=prompt,
                system_text="Rewrite queries for retrieval. Be minimal and precise.",
                max_tokens=120,
                temperature=0.0,
            ).strip()

            if not rewritten or len(rewritten) > 400:
                return query
            return rewritten
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query

    # -------------------------
    # Context prep
    # -------------------------
    def _prepare_context(
        self,
        chunks: List[Dict[str, Any]],
        similar_qa: List[Dict[str, Any]],
        session_id: str,
    ) -> str:
        parts: List[str] = []

        # A) Document chunks with explicit [pX] tags
        if chunks:
            parts.append("DOCUMENT EXCERPTS (use these to answer; cite like [p3]):")
            for ch in chunks:
                meta = ch.get("metadata", {}) or {}
                page = meta.get("page_num") or meta.get("page") or meta.get("page_number") or "?"
                doc_id = meta.get("doc_id", "unknown")[:8]
                content = (ch.get("content") or "").strip()
                if not content:
                    continue
                parts.append(f"[Doc: {doc_id}, p{page}] {content}")

        # B) Similar Q&A from memory
        if similar_qa:
            parts.append("\nRELATED PAST Q&A (use only if consistent with excerpts):")
            for qa in similar_qa[:3]:
                q = (qa.get("question") or "").strip()
                a = (qa.get("answer") or "").strip()
                if q and a:
                    parts.append(f"Q: {q}\nA: {a}")

        # C) Recent conversation for continuity
        short_term = self.memory_system.get_short_term_context(session_id) or []
        if short_term:
            parts.append("\nRECENT CONVERSATION (for tone only; do not invent facts):")
            for item in short_term[-2:]:
                parts.append(f"User: {item.get('query','')}")
                parts.append(f"Assistant: {item.get('answer','')}")

        return "\n".join(parts).strip()

    def _trim_context(self, text: str) -> str:
        if len(text) <= self.max_context_chars:
            return text
        return text[: self.max_context_chars] + "\n\n[Context trimmed]"

    def _groq_chat(
        self,
        model: str,
        user_text: str,
        system_text: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})

        last_err: Optional[Exception] = None
        for attempt in range(6):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 20))
        raise last_err  # type: ignore[misc]

    # -------------------------
    # Confidence
    # -------------------------
    def _calculate_confidence(self, answer: str, chunks: List[Dict[str, Any]]) -> float:
        if not chunks:
            return 0.0

        a = (answer or "").lower().strip()
        if "i can’t find" in a or "i can't find" in a or "not found in the provided documents" in a:
            return 0.15

        # citations boost
        citations = len(re.findall(r"\[p\d+\]", answer))
        cite_boost = min(citations / 4.0, 1.0)

        # prefer 'score' if available
        scores = [c.get("score") for c in chunks if isinstance(c.get("score"), (int, float))]
        if scores:
            dist_conf = sum(scores) / len(scores)
        else:
            distances = [c.get("distance") for c in chunks if isinstance(c.get("distance"), (int, float))]
            if distances:
                avg_d = sum(distances) / len(distances)
                dist_conf = 1.0 / (1.0 + avg_d)
            else:
                dist_conf = 0.5

        # answer length sanity
        len_conf = min(len(answer) / 400.0, 1.0)

        conf = (0.45 * dist_conf) + (0.35 * cite_boost) + (0.20 * len_conf)
        return float(max(0.0, min(conf, 1.0)))

    # -------------------------
    # Sources formatting
    # -------------------------
    def _format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        for ch in chunks:
            meta = ch.get("metadata", {}) or {}
            sources.append(
                {
                    "doc_id": meta.get("doc_id"),
                    "page_num": meta.get("page_num") or meta.get("page") or meta.get("page_number"),
                    "chunk_num": meta.get("chunk_num"),
                    "source": meta.get("source"),
                    "distance": ch.get("distance"),
                    "score": ch.get("score"),
                }
            )
        return sources

    # -------------------------
    # Cache
    # -------------------------
    def _generate_cache_key(self, query: str, session_id: str, document_filters: Optional[Dict[str, Any]]) -> str:
        filters_part = str(sorted((document_filters or {}).items()))
        content = f"{session_id}||{query}||{filters_part}||{self.answer_model}"
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        item = self.qa_cache.get(key)
        if not item:
            return None
        ts, data = item
        if (time.time() - ts) > self.cache_ttl_seconds:
            self.qa_cache.pop(key, None)
            return None
        return data

    def _cache_set(self, key: str, value: Dict[str, Any]) -> None:
        self.qa_cache[key] = (time.time(), value)

    # -------------------------
    # Feedback hooks
    # -------------------------
    def provide_feedback(
        self,
        interaction_id: str,
        feedback_type: str,
        feedback_data: Dict[str, Any],
        corrected_answer: Optional[str] = None,
    ):
        self.vector_db.store_feedback(interaction_id, feedback_type, feedback_data, corrected_answer)
        if corrected_answer:
            self._learn_from_correction(interaction_id, corrected_answer)

    def _learn_from_correction(self, interaction_id: str, corrected_answer: str):
        logger.info(f"Learning from correction for interaction {interaction_id}")

    # -------------------------
    # Topic extraction
    # -------------------------
    def _extract_topic(self, query: str) -> str:
        topics = ["technology", "science", "history", "business", "health", "education"]
        q = (query or "").lower()
        for t in topics:
            if t in q:
                return t
        return "general"
    