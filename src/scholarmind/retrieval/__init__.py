"""Persistent FAISS vector store, embedding cache, and hybrid dense+BM25 semantic retrieval."""

from scholarmind.retrieval.embedding_cache import (
    CachedTextEmbedder,
    get_shared_sentence_embedder,
)
from scholarmind.retrieval.hybrid_search import HybridSearchEngine
from scholarmind.retrieval.vector_store import FAISSVectorStore

__all__ = [
    "CachedTextEmbedder",
    "FAISSVectorStore",
    "HybridSearchEngine",
    "get_shared_sentence_embedder",
]
