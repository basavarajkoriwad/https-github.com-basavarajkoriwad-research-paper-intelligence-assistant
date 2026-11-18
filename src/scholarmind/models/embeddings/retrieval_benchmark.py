"""Experiment E: Comparing TF-IDF, BERT/SciBERT, and Sentence-Transformer Embeddings for Retrieval.

Mathematical Foundations:
1. TF-IDF (Lexical Sparse Vector Space):
       tfidf(t, d) = (1 + log tf(t, d)) * log((1 + N) / (1 + df(t)))
   Excels on exact scientific acronyms/terms but suffers from vocabulary mismatch (synonyms).

2. Pretrained BERT Embeddings (Mean vs. [CLS] Pooling):
       e_mean(d) = L2Norm( (1 / T) * sum_{i=1}^{T} h_i^(L) )
   Masked-LM pretraining optimizes token prediction rather than sentence-level cosine geometry,
   causing anisotropic embedding spaces where `[CLS]` often underperforms mean pooling.

3. Sentence Transformers (Bi-Encoder Contrastive / InfoNCE Objective):
       L_InfoNCE = -log( exp(sim(q, d^+) / tau) / sum_{j} exp(sim(q, d_j) / tau) )
   Directly aligns semantically matching (query, document) pairs on the unit hypersphere.

4. Retrieval Metrics (exact relevant document at rank r_i for query i):
       Recall@k = (1 / Q) * sum_{i=1}^{Q} 1[r_i <= k]
       MRR@k    = (1 / Q) * sum_{i=1}^{Q} (1 / r_i) * 1[r_i <= k]
       nDCG@k   = (1 / Q) * sum_{i=1}^{Q} (1 / log2(r_i + 1)) * 1[r_i <= k]
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scholarmind.core.logging import get_logger

logger = get_logger(__name__)


class BaseTextEmbedder(ABC):
    """Unified interface for sparse and dense text representation models."""

    name: str

    def fit_corpus(self, corpus: list[str]) -> None:
        """Optional corpus fitting step (used by TF-IDF; no-op for neural encoders)."""

    @abstractmethod
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Return L2-normalized 2D float32 array of shape `(len(texts), dim)`."""


class TFIDFEmbedder(BaseTextEmbedder):
    """Sublinear TF-IDF lexical baseline with unigram+bigram features."""

    def __init__(self, max_features: int = 25000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.name = "TF-IDF (1-2 gram)"
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            norm="l2",
        )

    def fit_corpus(self, corpus: list[str]) -> None:
        self.vectorizer.fit(corpus)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        mat = self.vectorizer.transform(texts)
        return mat.toarray().astype(np.float32)


class BERTContextualEmbedder(BaseTextEmbedder):
    """Extracts L2-normalized document vectors from a pretrained BERT / SciBERT model."""

    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        pooling: str = "mean",
        max_length: int = 256,
        batch_size: int = 16,
        device: torch.device | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        short = model_name.split("/")[-1]
        self.name = f"BERT ({short}, {pooling})"
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.model = (model or AutoModel.from_pretrained(model_name)).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state
            if self.pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            normed = F.normalize(pooled, p=2, dim=-1).cpu().numpy().astype(np.float32)
            chunks.append(normed)
        return np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)


class SentenceTransformerEmbedder(BaseTextEmbedder):
    """Contrastively trained dense bi-encoder via `sentence-transformers`."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        device: str = "cpu",
        query_instruction: str = "Represent this sentence for searching relevant passages: ",
        cache_folder: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.query_instruction = query_instruction if "bge" in model_name.lower() else ""
        short = model_name.split("/")[-1]
        self.name = f"SentenceTransformer ({short})"
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_folder,
            local_files_only=local_files_only,
        )

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        inputs = (
            [self.query_instruction + t for t in texts]
            if (is_query and self.query_instruction)
            else texts
        )
        vecs = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)


def evaluate_retrieval(
    embedder: BaseTextEmbedder,
    queries: list[str],
    corpus: list[str],
    gold_doc_indices: list[int],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Benchmark an embedder on exact scientific retrieval using cosine similarity."""
    t_fit0 = time.perf_counter()
    embedder.fit_corpus(corpus)
    corpus_vecs = embedder.encode(corpus, is_query=False)
    corpus_encode_sec = time.perf_counter() - t_fit0

    t_q0 = time.perf_counter()
    query_vecs = embedder.encode(queries, is_query=True)
    sim_matrix = query_vecs @ corpus_vecs.T  # Both L2-normalized -> exact cosine similarity
    query_elapsed_sec = time.perf_counter() - t_q0

    n_queries = len(queries)
    recall_at = {1: 0.0, 5: 0.0, 10: 0.0}
    mrr_10 = 0.0
    ndcg_10 = 0.0

    for q_idx, gold_idx in enumerate(gold_doc_indices):
        scores = sim_matrix[q_idx]
        # Rank descending
        ranked = np.argsort(-scores)
        pos = int(np.where(ranked == gold_idx)[0][0]) + 1  # 1-based rank

        for k in (1, 5, 10):
            if pos <= k:
                recall_at[k] += 1.0
        if pos <= 10:
            mrr_10 += 1.0 / pos
            ndcg_10 += 1.0 / math.log2(pos + 1.0)

    denom = max(1, n_queries)
    metrics = {
        "model": embedder.name,
        "embedding_dim": int(corpus_vecs.shape[1]) if corpus_vecs.ndim == 2 else 0,
        "num_queries": n_queries,
        "corpus_size": len(corpus),
        "recall_at_1": round(recall_at[1] / denom, 4),
        "recall_at_5": round(recall_at[5] / denom, 4),
        "recall_at_10": round(recall_at[10] / denom, 4),
        "mrr_at_10": round(mrr_10 / denom, 4),
        "ndcg_at_10": round(ndcg_10 / denom, 4),
        "corpus_encode_seconds": round(corpus_encode_sec, 3),
        "query_latency_ms": round((query_elapsed_sec / denom) * 1000.0, 3),
    }
    return metrics, query_vecs, corpus_vecs


def plot_retrieval_comparison(
    benchmark_results: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> Path:
    """Render side-by-side grouped bar charts comparing embedding paradigms across tasks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_tasks = len(benchmark_results)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6.8 * n_tasks, 4.8))
    if n_tasks == 1:
        axes = [axes]

    metrics_to_plot = ["recall_at_1", "recall_at_5", "mrr_at_10", "ndcg_at_10"]
    metric_labels = ["Recall@1", "Recall@5", "MRR@10", "nDCG@10"]

    for ax, (task_name, rows) in zip(axes, benchmark_results.items()):
        x = np.arange(len(metrics_to_plot))
        width = 0.78 / max(1, len(rows))
        for i, row in enumerate(rows):
            vals = [row[m] for m in metrics_to_plot]
            offset = (i - (len(rows) - 1) / 2.0) * width
            ax.bar(x + offset, vals, width=width, label=row["model"])

        ax.set_title(task_name)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
