"""Embedding models and scientific retrieval evaluation benchmarks."""

from scholarmind.models.embeddings.retrieval_benchmark import (
    BaseTextEmbedder,
    BERTContextualEmbedder,
    SentenceTransformerEmbedder,
    TFIDFEmbedder,
    evaluate_retrieval,
    plot_retrieval_comparison,
)

__all__ = [
    "BERTContextualEmbedder",
    "BaseTextEmbedder",
    "SentenceTransformerEmbedder",
    "TFIDFEmbedder",
    "evaluate_retrieval",
    "plot_retrieval_comparison",
]
