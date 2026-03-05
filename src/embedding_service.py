# src/embedding_service.py
from typing import List, Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Local embeddings (recommended for RAG):
    - No API quota issues
    - Faster for repeated runs
    - Stable dimensions
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            # infer dimension once
            self.dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.exception("Failed to load SentenceTransformer model. Install: pip install sentence-transformers")
            raise e

    def generate_embeddings(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """
        Generate embeddings in batches.
        """
        if not texts:
            return []

        cleaned = [self._clean_text(t) for t in texts]
        vectors = []

        for i in range(0, len(cleaned), self.batch_size):
            batch = cleaned[i:i + self.batch_size]
            emb = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            vectors.extend(emb.tolist())

        return vectors

    def generate_hierarchical_embeddings(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates:
        - doc embedding (full doc)
        - section embeddings (simple grouping)
        - chunk embeddings
        """
        if not chunks:
            return {"document": [], "sections": {}, "chunks": []}

        document_text = " ".join([c.get("content", "") for c in chunks]).strip()
        doc_embedding = self.generate_embeddings([document_text])[0] if document_text else [0.0] * self.dim

        sections = self._group_into_sections(chunks)
        section_embeddings = {}
        for section_id, section_chunks in sections.items():
            section_text = " ".join([c.get("content", "") for c in section_chunks]).strip()
            if section_text:
                section_embeddings[section_id] = self.generate_embeddings([section_text])[0]
            else:
                section_embeddings[section_id] = [0.0] * self.dim

        chunk_texts = [c.get("content", "") for c in chunks]
        chunk_embeddings = self.generate_embeddings(chunk_texts)

        return {"document": doc_embedding, "sections": section_embeddings, "chunks": chunk_embeddings}

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Cosine similarity. If embeddings are normalized, dot product is enough.
        """
        v1 = np.array(embedding1, dtype=np.float32)
        v2 = np.array(embedding2, dtype=np.float32)

        if v1.size == 0 or v2.size == 0:
            return 0.0

        # works for normalized or not
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0:
            return 0.0
        return float(np.dot(v1, v2) / denom)

    def _clean_text(self, text: str) -> str:
        return " ".join((text or "").split())

    def _group_into_sections(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Better than keyword-search: use metadata if available.
        If your DocumentProcessor (Unstructured) stores titles/headings in metadata,
        group by that. Else fallback to a simple fixed-size grouping.
        """
        sections: Dict[str, List[Dict[str, Any]]] = {}

        # If chunk metadata contains 'section' or 'title', group by it
        for c in chunks:
            meta = c.get("metadata", {}) if isinstance(c, dict) else {}
            key = meta.get("section") or meta.get("title") or meta.get("heading")

            if not key:
                key = "section_default"

            sections.setdefault(str(key), []).append(c)

        # If everything went to one bucket, optionally split into blocks of N chunks
        if len(sections) == 1:
            only_key = next(iter(sections.keys()))
            items = sections[only_key]
            if len(items) > 20:
                sections = {}
                block = 20
                for i in range(0, len(items), block):
                    sections[f"section_{i//block + 1}"] = items[i:i+block]

        return sections