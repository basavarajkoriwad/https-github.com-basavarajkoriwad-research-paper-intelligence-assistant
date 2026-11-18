"""Experiment D: T5 Transformer Summarization on ArXiv Summarization + ROUGE Evaluation."""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scholarmind.core.config import AppConfig, load_config
from scholarmind.core.device import get_device, set_seed
from scholarmind.core.logging import get_logger
from scholarmind.data.storage import load_parquet
from scholarmind.models.t5.t5_summarizer import (
    T5ScientificSummarizer,
    compute_rouge_metrics,
    lead_n_baseline,
)

logger = get_logger(__name__)


def run_experiment_d(
    config: AppConfig | None = None,
    model_name: str = "google/flan-t5-small",
    max_train_samples: int = 48,
    max_val_samples: int = 16,
    max_test_samples: int = 16,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute Experiment D: Lead-3 vs. Pretrained T5 vs. Hierarchical T5 vs. Fine-tuned T5."""
    cfg = config or load_config()
    set_seed(seed)
    device = get_device(cfg.hardware.get("preferred_device", "auto"))

    ds_dir = cfg.processed_dir / "arxiv_summarization"
    train_df = load_parquet(ds_dir / "train.parquet").head(max_train_samples).reset_index(drop=True)
    val_df = load_parquet(ds_dir / "val.parquet").head(max_val_samples).reset_index(drop=True)
    test_df = load_parquet(ds_dir / "test.parquet").head(max_test_samples).reset_index(drop=True)

    test_articles = test_df["article_clean"].astype(str).tolist()
    test_refs = test_df["abstract_clean"].astype(str).tolist()

    logger.info(
        "Experiment D (T5: %s) | train=%d, val=%d, test=%d, device=%s",
        model_name,
        len(train_df),
        len(val_df),
        len(test_df),
        device,
    )

    # 1. Extractive Lead-3 Baseline
    lead3_preds = lead_n_baseline(test_articles, n_sentences=3)
    lead3_rouge = compute_rouge_metrics(lead3_preds, test_refs)
    logger.info("Lead-3 Baseline ROUGE: %s", lead3_rouge)

    # 2. Pretrained Flan-T5 Zero-Shot Single-Pass Inference
    summarizer = T5ScientificSummarizer(model_name=model_name, device=device)
    pretrained_preds = summarizer.summarize_batch(test_articles, batch_size=batch_size, num_beams=2)
    pretrained_rouge = compute_rouge_metrics(pretrained_preds, test_refs)
    logger.info("Pretrained T5 Single-Pass ROUGE: %s", pretrained_rouge)

    # 3. Pretrained Flan-T5 Hierarchical Map-Reduce on a subset (first 6 test papers for CPU speed)
    mr_n = min(6, len(test_articles))
    mr_preds = [
        summarizer.summarize_hierarchical_map_reduce(art, max_chunks=3, chunk_words=260)
        for art in test_articles[:mr_n]
    ]
    mr_rouge = compute_rouge_metrics(mr_preds, test_refs[:mr_n])
    logger.info("Pretrained T5 Hierarchical Map-Reduce (n=%d) ROUGE: %s", mr_n, mr_rouge)

    # 4. PyTorch Fine-Tuning on ArXiv Summarization train split
    ckpt_path = cfg.artifacts_dir / "checkpoints" / "exp_d_t5" / "best_model.pt"
    ft_res = summarizer.fine_tune(
        train_df=train_df,
        val_df=val_df,
        checkpoint_path=ckpt_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_source_len=384,
        max_target_len=112,
    )

    # 5. Evaluate Fine-Tuned T5 on the exact same held-out test papers
    finetuned_preds = summarizer.summarize_batch(test_articles, batch_size=batch_size, num_beams=2)
    finetuned_rouge = compute_rouge_metrics(finetuned_preds, test_refs)
    logger.info("Fine-Tuned T5 ROUGE: %s", finetuned_rouge)

    # Plot T5 fine-tuning loss curve
    plot_path = cfg.reports_dir / "plots" / "exp_d_t5_curves.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    hist = ft_res["history"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ep = [h["epoch"] for h in hist]
    ax.plot(ep, [h["train_loss"] for h in hist], marker="o", label="Train Loss")
    ax.plot(ep, [h["val_loss"] for h in hist], marker="s", linestyle="--", label="Val Loss")
    ax.axhline(
        ft_res["pretrained_val_loss"],
        color="gray",
        linestyle=":",
        label=f"Pretrained Val Loss ({ft_res['pretrained_val_loss']})",
    )
    ax.set_title(f"Experiment D: {model_name} Fine-Tuning on ArXiv Summarization")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    sample_comparison = [
        {
            "paper_id": str(test_df.iloc[i]["paper_id"]),
            "reference_abstract": test_refs[i][:350],
            "pretrained_summary": pretrained_preds[i],
            "finetuned_summary": finetuned_preds[i],
        }
        for i in range(min(2, len(test_refs)))
    ]

    exp_dir = cfg.artifacts_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "Experiment D: T5 Scientific Paper Summarization",
        "dataset": "arxiv_summarization",
        "model_name": model_name,
        "seed": seed,
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "lead3_baseline_rouge": lead3_rouge,
        "pretrained_t5_rouge": pretrained_rouge,
        "hierarchical_map_reduce_rouge": {"num_papers": mr_n, **mr_rouge},
        "finetuned_t5_rouge": finetuned_rouge,
        "finetuning": ft_res,
        "sample_summaries": sample_comparison,
        "loss_curve_plot": str(plot_path),
    }

    out_json = exp_dir / "exp_d_t5_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved Experiment D results -> %s", out_json)
    return payload


if __name__ == "__main__":
    run_experiment_d()
