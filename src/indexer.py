# src/indexer.py
from __future__ import annotations

import hashlib
import logging
from typing import Optional, Dict, Any, List, Tuple

from .document_processor import DocumentProcessor, DocumentChunk
from .embedding_service import EmbeddingService
from .vector_db import VectorDatabase

logger = logging.getLogger(__name__)


def index_document(
    file_path: str,
    doc_id: str,
    processor: DocumentProcessor,
    embedder: EmbeddingService,
    vector_db: VectorDatabase,
    extra_metadata: Optional[Dict[str, Any]] = None,
    min_chunk_chars: int = 30,
) -> Dict[str, Any]:
    """
    Loads a document -> chunks -> embeddings -> stores in Chroma using UPSERT.

    Improvements vs simple version:
    - Filters empty/too-small chunks
    - Attaches doc_id + filename to metadata consistently (helps filtering)
    - Embeds in batch (EmbeddingService already batches)
    - Returns stats for logging/monitoring
    """

    extra_metadata = extra_metadata or {}

    # 1) Chunk the document
    chunks: List[DocumentChunk] = processor.process_document(file_path, doc_id)

    # 2) Convert to dicts + clean content + enforce metadata
    chunk_dicts: List[Dict[str, Any]] = []
    for c in chunks:
        text = (c.content or "").strip()
        if len(text) < min_chunk_chars:
            continue

        meta = dict(c.metadata or {})
        meta.update(extra_metadata)

        # Ensure consistent keys (good for where filters later)
        meta["doc_id"] = doc_id
        meta.setdefault("source_file", file_path)

        chunk_dicts.append(
            {
                "chunk_id": c.chunk_id,
                "content": text,
                "metadata": meta,
            }
        )

    if not chunk_dicts:
        return {
            "doc_id": doc_id,
            "file_path": file_path,
            "chunks_total": len(chunks),
            "chunks_indexed": 0,
            "status": "no_valid_chunks",
        }

    # 3) Embed all chunks (EmbeddingService should batch internally)
    texts = [c["content"] for c in chunk_dicts]
    embeddings = embedder.generate_embeddings(texts)

    # 4) UPSERT into Chroma
    vector_db.upsert_document_chunks(chunk_dicts, embeddings)

    return {
        "doc_id": doc_id,
        "file_path": file_path,
        "chunks_total": len(chunks),
        "chunks_indexed": len(chunk_dicts),
        "embedding_dim": len(embeddings[0]) if embeddings else None,
        "status": "indexed",
    }