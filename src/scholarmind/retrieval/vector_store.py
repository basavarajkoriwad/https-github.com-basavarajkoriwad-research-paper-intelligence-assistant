"""Persistent FAISS cosine-similarity vector index with embedding caching and document sync."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from scholarmind.core.logging import get_logger
from scholarmind.core.schemas import DocumentChunk, RetrievedPassage
from scholarmind.models.embeddings.retrieval_benchmark import BaseTextEmbedder
from scholarmind.retrieval.embedding_cache import (
    CachedTextEmbedder,
    get_shared_sentence_embedder,
)

logger = get_logger(__name__)


class FAISSVectorStore:
    """Disk-persisted FAISS `IndexFlatIP` store over L2-normalized Sentence Transformer vectors."""

    def __init__(
        self,
        index_dir: Path,
        embedder: BaseTextEmbedder | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hf_cache_dir: Path | None = None,
        enable_embedding_cache: bool = True,
    ) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "chunks.faiss"
        self.meta_path = self.index_dir / "chunks_meta.json"
        self.model_name = model_name
        self.hf_cache_dir = hf_cache_dir
        self.enable_embedding_cache = enable_embedding_cache

        self._embedder: BaseTextEmbedder | None = embedder
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []
        self._load_if_exists()

    @property
    def embedder(self) -> BaseTextEmbedder:
        """Lazy-load shared SentenceTransformer wrapped with persistent SHA-256 embedding cache."""
        if self._embedder is None:
            base = get_shared_sentence_embedder(
                model_name=self.model_name,
                hf_cache_dir=self.hf_cache_dir,
                device="auto",
            )
            self._embedder = (
                CachedTextEmbedder(base, cache_dir=self.index_dir / "emb_cache")
                if self.enable_embedding_cache
                else base
            )
        return self._embedder

    @property
    def size(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0

    def has_paper(self, paper_id: str) -> bool:
        """Return True if chunks for `paper_id` are already indexed in FAISS."""
        return any(m.get("paper_id") == paper_id for m in self.metadata)

    def _load_if_exists(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if self.index.ntotal != len(self.metadata):
                logger.warning(
                    "FAISS index count (%d) != metadata count (%d); index may require rebuild.",
                    self.index.ntotal,
                    len(self.metadata),
                )
            logger.info("Loaded persisted FAISS index (%d vectors) from %s.", self.size, self.index_dir)

    def _persist(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        elif self.index_path.exists():
            self.index_path.unlink()
        self.meta_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Embed and index new `DocumentChunk` records, deduplicating any existing paper chunks."""
        if not chunks:
            return 0

        incoming_paper_ids = {c.paper_id for c in chunks}
        if any(m["paper_id"] in incoming_paper_ids for m in self.metadata):
            remaining_meta = [m for m in self.metadata if m["paper_id"] not in incoming_paper_ids]
            existing_chunks = [
                DocumentChunk(
                    chunk_id=m["chunk_id"],
                    paper_id=m["paper_id"],
                    document_name=m.get("document_name", ""),
                    paper_title=m["paper_title"],
                    section_title=m["section_title"],
                    section_type=m["section_type"],
                    page_number=m["page_number"],
                    text=m["text"],
                    contextualized_text=m["contextualized_text"],
                )
                for m in remaining_meta
            ]
            self.rebuild(existing_chunks + chunks)
            return len(chunks)

        texts = [c.contextualized_text for c in chunks]
        vectors = self.embedder.encode(texts, is_query=False)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            return 0

        # Ensure unit L2 norm so inner product equals exact cosine similarity
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
        elif self.index.d != dim:
            raise ValueError(
                f"Embedding dim mismatch: existing FAISS index has d={self.index.d}, got d={dim}."
            )

        self.index.add(vectors)
        for c in chunks:
            self.metadata.append(
                {
                    "chunk_id": c.chunk_id,
                    "paper_id": c.paper_id,
                    "document_name": c.document_name or c.paper_title,
                    "paper_title": c.paper_title,
                    "section_title": c.section_title,
                    "section_type": c.section_type,
                    "page_number": c.page_number,
                    "text": c.text,
                    "contextualized_text": c.contextualized_text,
                }
            )
        self._persist()
        logger.info("Indexed %d chunks -> FAISS total=%d vectors.", len(chunks), self.size)
        return len(chunks)

    def rebuild(self, all_chunks: list[DocumentChunk]) -> int:
        """Rebuild the entire FAISS index cleanly from the canonical chunk list."""
        self.index = None
        self.metadata = []
        if not all_chunks:
            self._persist()
            return 0
        return self.add_chunks(all_chunks)

    def delete_paper(self, paper_id: str) -> int:
        """Remove all vectors belonging to `paper_id` and persist the synchronized FAISS index."""
        remaining_meta = [m for m in self.metadata if m["paper_id"] != paper_id]
        removed = len(self.metadata) - len(remaining_meta)
        if removed == 0:
            return 0
        remaining_chunks = [
            DocumentChunk(
                chunk_id=m["chunk_id"],
                paper_id=m["paper_id"],
                document_name=m.get("document_name", ""),
                paper_title=m["paper_title"],
                section_title=m["section_title"],
                section_type=m["section_type"],
                page_number=m["page_number"],
                text=m["text"],
                contextualized_text=m["contextualized_text"],
            )
            for m in remaining_meta
        ]
        self.rebuild(remaining_chunks)
        return removed

    def search(
        self,
        query: str,
        top_k: int = 5,
        paper_ids: list[str] | None = None,
        section_types: list[str] | None = None,
    ) -> list[RetrievedPassage]:
        """Search FAISS by cosine similarity with optional document and section filtering."""
        if self.index is None or self.size == 0 or not query.strip():
            return []

        q_vec = self.embedder.encode([query], is_query=True)
        q_vec = np.ascontiguousarray(q_vec, dtype=np.float32)
        faiss.normalize_L2(q_vec)

        fetch_k = self.size if (paper_ids or section_types) else min(self.size, max(top_k * 3, top_k))
        scores, indices = self.index.search(q_vec, fetch_k)

        allowed_papers = set(paper_ids) if paper_ids else None
        allowed_sections = {s.lower() for s in section_types} if section_types else None

        results: list[RetrievedPassage] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            if allowed_papers is not None and meta["paper_id"] not in allowed_papers:
                continue
            if allowed_sections is not None and meta["section_type"].lower() not in allowed_sections:
                continue

            results.append(
                RetrievedPassage(
                    rank=len(results) + 1,
                    chunk_id=meta["chunk_id"],
                    paper_id=meta["paper_id"],
                    document_name=meta.get("document_name") or meta["paper_title"],
                    paper_title=meta["paper_title"],
                    section_title=meta["section_title"],
                    section_type=meta["section_type"],
                    page_number=int(meta["page_number"]),
                    score=round(float(score), 4),
                    dense_score=round(float(score), 4),
                    lexical_score=0.0,
                    text=meta["text"],
                )
            )
            if len(results) >= top_k:
                break

        return results
