"""Experiment E: Comparing TF-IDF, BERT/SciBERT, and Sentence Transformers for Scientific Retrieval."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from scholarmind.core.config import AppConfig, load_config
from scholarmind.core.device import get_device, set_seed
from scholarmind.core.logging import get_logger
from scholarmind.data.storage import load_parquet
from scholarmind.models.embeddings.retrieval_benchmark import (
    BERTContextualEmbedder,
    SentenceTransformerEmbedder,
    TFIDFEmbedder,
    evaluate_retrieval,
    plot_retrieval_comparison,
)

logger = get_logger(__name__)


def run_experiment_e(
    config: AppConfig | None = None,
    bert_model_name: str = "allenai/scibert_scivocab_uncased",
    st_model_name: str = "BAAI/bge-small-en-v1.5",
    seed: int = 42,
) -> dict[str, Any]:
    """Benchmark TF-IDF vs. BERT ([CLS] & Mean) vs. Sentence Transformers on 2 scientific retrieval tasks."""
    cfg = config or load_config()
    set_seed(seed)
    device = get_device(cfg.hardware.get("preferred_device", "auto"))

    # Task 1: ArXiv Title -> Abstract Retrieval (held-out val + test papers)
    arxiv_dir = cfg.processed_dir / "arxiv_metadata"
    arxiv_eval = pd.concat(
        [load_parquet(arxiv_dir / "val.parquet"), load_parquet(arxiv_dir / "test.parquet")],
        ignore_index=True,
    )
    arxiv_queries = arxiv_eval["title_clean"].astype(str).tolist()
    arxiv_corpus = arxiv_eval["abstract_clean"].astype(str).tolist()
    arxiv_gold = list(range(len(arxiv_eval)))

    # Task 2: SciQ Question -> Supporting Scientific Passage Retrieval (held-out test split with support)
    sciq_test = load_parquet(cfg.processed_dir / "sciq" / "test.parquet")
    sciq_eval = sciq_test[sciq_test["has_support"]].reset_index(drop=True)
    sciq_queries = sciq_eval["question_clean"].astype(str).tolist()
    sciq_corpus = sciq_eval["support_clean"].astype(str).tolist()
    sciq_gold = list(range(len(sciq_eval)))

    logger.info(
        "Experiment E (Embeddings) | ArXiv queries=%d | SciQ queries=%d | device=%s",
        len(arxiv_queries),
        len(sciq_queries),
        device,
    )

    # Share a single loaded SciBERT backbone between Mean and [CLS] pooling to save RAM/time
    bert_mean = BERTContextualEmbedder(model_name=bert_model_name, pooling="mean", device=device)
    bert_cls = BERTContextualEmbedder(
        model_name=bert_model_name,
        pooling="cls",
        device=device,
        tokenizer=bert_mean.tokenizer,
        model=bert_mean.model,
    )
    st_embedder = SentenceTransformerEmbedder(model_name=st_model_name, device=str(device))

    embedders = [
        TFIDFEmbedder(),
        bert_cls,
        bert_mean,
        st_embedder,
    ]

    arxiv_results: list[dict[str, Any]] = []
    sciq_results: list[dict[str, Any]] = []
    saved_arrays: dict[str, np.ndarray] = {}

    for emb in embedders:
        res_arxiv, _, arxiv_cvecs = evaluate_retrieval(emb, arxiv_queries, arxiv_corpus, arxiv_gold)
        arxiv_results.append(res_arxiv)
        logger.info("ArXiv Retrieval | %s -> %s", emb.name, res_arxiv)

        res_sciq, _, sciq_cvecs = evaluate_retrieval(emb, sciq_queries, sciq_corpus, sciq_gold)
        sciq_results.append(res_sciq)
        logger.info("SciQ Retrieval  | %s -> %s", emb.name, res_sciq)

        # Save dense neural embedding checkpoint matrices for downstream inspection
        if not isinstance(emb, TFIDFEmbedder):
            safe_key = emb.name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("-", "_")
            saved_arrays[f"arxiv_{safe_key}"] = arxiv_cvecs
            saved_arrays[f"sciq_{safe_key}"] = sciq_cvecs

    ckpt_dir = cfg.artifacts_dir / "checkpoints" / "exp_e_embeddings"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / "scientific_embeddings_checkpoint.npz"
    np.savez_compressed(ckpt_file, **saved_arrays)

    plot_path = plot_retrieval_comparison(
        {
            "Task 1: ArXiv Title -> Abstract Retrieval": arxiv_results,
            "Task 2: SciQ Question -> Support Passage Retrieval": sciq_results,
        },
        output_path=cfg.reports_dir / "plots" / "exp_e_embeddings_comparison.png",
    )

    exp_dir = cfg.artifacts_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "Experiment E: Embedding Comparison for Scientific Retrieval",
        "seed": seed,
        "bert_model": bert_model_name,
        "sentence_transformer_model": st_model_name,
        "arxiv_title_to_abstract": arxiv_results,
        "sciq_question_to_support": sciq_results,
        "checkpoint_path": str(ckpt_file),
        "comparison_plot": str(plot_path),
    }

    out_json = exp_dir / "exp_e_embeddings_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved Experiment E results -> %s", out_json)
    return payload


if __name__ == "__main__":
    run_experiment_e()
