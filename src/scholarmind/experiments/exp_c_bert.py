"""Experiment C: Pretrained SciBERT Contextual Representations & Fine-Tuning."""

from __future__ import annotations

import json
from typing import Any

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from scholarmind.core.config import AppConfig, load_config
from scholarmind.core.device import get_device, set_seed
from scholarmind.core.logging import get_logger
from scholarmind.data.storage import load_parquet
from scholarmind.models.bert.sci_bert_classifier import (
    ArxivBERTDataset,
    SciBERTClassifier,
    demonstrate_contextual_embeddings,
    train_bert_classifier,
)
from scholarmind.models.rnn_lstm.dataset import build_label_mapping
from scholarmind.models.rnn_lstm.trainer import plot_loss_curves

logger = get_logger(__name__)


def run_experiment_c(
    config: AppConfig | None = None,
    model_name: str = "allenai/scibert_scivocab_uncased",
    epochs: int = 4,
    batch_size: int = 16,
    max_length: int = 192,
    unfreeze_last_n_layers: int = 2,
    lr: float = 5e-5,
    head_lr: float = 5e-4,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute Experiment C: SciBERT contextual embedding analysis + ArXiv fine-tuning."""
    cfg = config or load_config()
    set_seed(seed)
    device = get_device(cfg.hardware.get("preferred_device", "auto"))

    ds_dir = cfg.processed_dir / "arxiv_metadata"
    train_df = load_parquet(ds_dir / "train.parquet")
    val_df = load_parquet(ds_dir / "val.parquet")
    test_df = load_parquet(ds_dir / "test.parquet")
    label_to_id, id_to_label = build_label_mapping(train_df)

    logger.info(
        "Experiment C (BERT: %s) | train=%d, val=%d, test=%d, unfreeze_last_n=%d, device=%s",
        model_name,
        len(train_df),
        len(val_df),
        len(test_df),
        unfreeze_last_n_layers,
        device,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SciBERTClassifier(
        model_name=model_name,
        num_classes=len(label_to_id),
        dropout=0.25,
        pooling="mean",
        unfreeze_last_n_layers=unfreeze_last_n_layers,
    )

    # 1. Contextual embedding probe BEFORE fine-tuning
    contextual_probe = demonstrate_contextual_embeddings(
        model_name=model_name,
        device=device,
        tokenizer=tokenizer,
        encoder=model.encoder,
    )

    # 2. Fine-tuning on the exact same ArXiv splits as Experiments A & B
    train_ds = ArxivBERTDataset(train_df, tokenizer, label_to_id, max_length=max_length)
    val_ds = ArxivBERTDataset(val_df, tokenizer, label_to_id, max_length=max_length)
    test_ds = ArxivBERTDataset(test_df, tokenizer, label_to_id, max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    ckpt_path = cfg.artifacts_dir / "checkpoints" / "exp_c_bert" / "best_model.pt"
    exp_dir = cfg.artifacts_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_res = train_bert_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        checkpoint_path=ckpt_path,
        id_to_label=id_to_label,
        epochs=epochs,
        lr=lr,
        head_lr=head_lr,
    )

    plot_path = plot_loss_curves(
        {"SciBERT (top-2 layers + head)": train_res["history"]},
        output_path=cfg.reports_dir / "plots" / "exp_c_bert_curves.png",
        title="Experiment C: SciBERT Fine-Tuning on ArXiv Classification",
    )

    payload = {
        "experiment": "Experiment C: Pretrained BERT (SciBERT) Representations & Fine-Tuning",
        "dataset": "arxiv_metadata",
        "seed": seed,
        "contextual_embedding_probe": contextual_probe,
        "scibert_finetune": train_res,
        "loss_curve_plot": str(plot_path),
    }

    out_json = exp_dir / "exp_c_bert_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved Experiment C results -> %s", out_json)
    return payload


if __name__ == "__main__":
    run_experiment_c()
