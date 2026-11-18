"""System status, model checkpoint availability, and saved experiment results routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from scholarmind import __version__
from scholarmind.api.dependencies import ServiceContainer, get_container
from scholarmind.core.device import get_device

router = APIRouter(tags=["System & Experiments"])


def _check_hf_cached(hf_cache_dir: Path, repo_id: str) -> bool:
    hub_dir = hf_cache_dir / "hub"
    folder_name = "models--" + repo_id.replace("/", "--")
    return (hub_dir / folder_name).exists()


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sanitize_paths(obj: Any) -> Any:
    """Strip absolute local paths from serialized experiment JSONs before returning via API."""
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            if k in ("checkpoint_path", "loss_curve_plot", "comparison_plot") and isinstance(v, str):
                cleaned[k] = Path(v).name
            else:
                cleaned[k] = _sanitize_paths(v)
        return cleaned
    if isinstance(obj, list):
        return [_sanitize_paths(x) for x in obj]
    return obj


@router.get("/system/status")
@router.get("/api/v1/system/status")
def system_status(container: ServiceContainer = Depends(get_container)) -> dict[str, Any]:
    """Return live dashboard telemetry: documents, FAISS index, Q&A stats, and real model availability."""
    cfg = container.config
    papers = container.document_store.list_papers()
    qa_stats = container.document_store.get_qa_stats(recent_limit=8)
    ckpt_root = cfg.artifacts_dir / "checkpoints"
    hf_cache = cfg.artifacts_dir / "hf_cache"

    models_inventory = [
        {
            "component": "Experiment A: Vanilla RNN",
            "model_id": "VanillaRNNClassifier (PyTorch)",
            "training_status": "Trained from scratch (ArXiv)",
            "available": (ckpt_root / "exp_a_rnn" / "best_model.pt").exists(),
        },
        {
            "component": "Experiment B: LSTM & BiLSTM+Attn",
            "model_id": "LSTM / BiLSTMAttentionClassifier (PyTorch)",
            "training_status": "Trained from scratch (ArXiv)",
            "available": (ckpt_root / "exp_b_lstm" / "best_model.pt").exists()
            and (ckpt_root / "exp_b_bilstm_attn" / "best_model.pt").exists(),
        },
        {
            "component": "Experiment C: SciBERT Classifier",
            "model_id": "allenai/scibert_scivocab_uncased",
            "training_status": "Fine-tuned top-2 layers + head (ArXiv)",
            "available": (ckpt_root / "exp_c_bert" / "best_model.pt").exists(),
        },
        {
            "component": "Experiment D: T5 Summarizer",
            "model_id": "google/flan-t5-small",
            "training_status": "Fine-tuned in PyTorch (ArXiv Summarization)",
            "available": (ckpt_root / "exp_d_t5" / "best_model.pt").exists(),
        },
        {
            "component": "RAG Embedding Model",
            "model_id": container.vector_store.model_name,
            "training_status": "Pretrained inference (cached locally)",
            "available": _check_hf_cached(hf_cache, container.vector_store.model_name),
        },
        {
            "component": "RAG Instruction LLM",
            "model_id": container.llm_client.backend_name,
            "training_status": "Pretrained open-source inference",
            "available": _check_hf_cached(hf_cache, "google/flan-t5-base")
            or container.llm_client.backend_name.startswith("ollama:"),
        },
    ]

    return {
        "service": "ScholarMind AI",
        "version": __version__,
        "profile": cfg.profile,
        "compute_device": str(get_device("auto")),
        "indexed_papers": len(papers),
        "total_pages": sum(int(p.get("page_count", 0)) for p in papers),
        "total_chunks": sum(int(p.get("chunk_count", 0)) for p in papers),
        "indexed_vectors": container.vector_store.size,
        "embedding_model": container.vector_store.model_name,
        "llm_backend": container.llm_client.backend_name,
        "qa_stats": qa_stats,
        "models_inventory": models_inventory,
        "papers": [
            {
                "document_id": p["paper_id"],
                "document_name": p["filename"],
                "title": p["title"],
                "page_count": p["page_count"],
                "word_count": p["word_count"],
                "chunk_count": p["chunk_count"],
                "ingested_at_utc": p["ingested_at_utc"],
            }
            for p in papers
        ],
    }


@router.get("/experiments/results")
@router.get("/api/v1/experiments/results")
def get_experiment_results(container: ServiceContainer = Depends(get_container)) -> dict[str, Any]:
    """Return actual saved experiment and evaluation JSON results (sanitized of local file paths)."""
    exp_dir = container.config.artifacts_dir / "experiments"
    reports_dir = container.config.reports_dir

    files = {
        "exp_a_rnn": exp_dir / "exp_a_rnn_results.json",
        "exp_b_lstm_bilstm": exp_dir / "exp_b_lstm_bilstm_results.json",
        "exp_c_bert": exp_dir / "exp_c_bert_results.json",
        "exp_d_t5": exp_dir / "exp_d_t5_results.json",
        "exp_e_embeddings": exp_dir / "exp_e_embeddings_results.json",
        "module5_rag_eval": exp_dir / "module5_rag_eval.json",
        "eda_summary": reports_dir / "eda_summary.json",
    }

    payload: dict[str, Any] = {}
    for key, path in files.items():
        loaded = _safe_load_json(path)
        if loaded is not None:
            payload[key] = _sanitize_paths(loaded)
    return payload
