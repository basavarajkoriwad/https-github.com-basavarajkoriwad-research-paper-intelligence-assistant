"""Experiment A: Vanilla RNN on ArXiv Scientific Domain Classification."""

from __future__ import annotations

import json
from functools import partial
from typing import Any

from torch.utils.data import DataLoader

from scholarmind.core.config import AppConfig, load_config
from scholarmind.core.device import get_device, set_seed
from scholarmind.core.logging import get_logger
from scholarmind.data.storage import load_parquet
from scholarmind.models.rnn_lstm.dataset import (
    ArxivSequenceDataset,
    build_label_mapping,
    collate_padded_sequences,
    compute_classical_baselines,
)
from scholarmind.models.rnn_lstm.trainer import plot_loss_curves, train_sequence_classifier
from scholarmind.models.rnn_lstm.vanilla_rnn import VanillaRNNClassifier
from scholarmind.nlp.tokenization import Vocabulary

logger = get_logger(__name__)


def run_experiment_a(
    config: AppConfig | None = None,
    epochs: int = 12,
    batch_size: int = 32,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 1,
    lr: float = 2e-3,
    max_len: int = 256,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute Experiment A (Vanilla RNN vs. Classical Baselines) on ArXiv Metadata."""
    cfg = config or load_config()
    set_seed(seed)
    device = get_device(cfg.hardware.get("preferred_device", "auto"))

    ds_dir = cfg.processed_dir / "arxiv_metadata"
    train_df = load_parquet(ds_dir / "train.parquet")
    val_df = load_parquet(ds_dir / "val.parquet")
    test_df = load_parquet(ds_dir / "test.parquet")
    vocab = Vocabulary.load(ds_dir / "rnn_vocab.json")
    label_to_id, id_to_label = build_label_mapping(train_df)

    logger.info(
        "Experiment A (Vanilla RNN) | train=%d, val=%d, test=%d, vocab=%d, classes=%d, device=%s",
        len(train_df),
        len(val_df),
        len(test_df),
        len(vocab),
        len(label_to_id),
        device,
    )

    baselines = compute_classical_baselines(train_df, val_df, test_df, seed=seed)

    train_ds = ArxivSequenceDataset(train_df, vocab, label_to_id, max_len=max_len)
    val_ds = ArxivSequenceDataset(val_df, vocab, label_to_id, max_len=max_len)
    test_ds = ArxivSequenceDataset(test_df, vocab, label_to_id, max_len=max_len)

    collate = partial(collate_padded_sequences, pad_id=vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # Train both last-state Vanilla RNN (classic Elman) and mean-pooled Vanilla RNN
    # so we can empirically observe the vanishing gradient effect on last hidden state h_T!
    model_last = VanillaRNNClassifier(
        vocab_size=len(vocab),
        num_classes=len(label_to_id),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.3,
        pad_idx=vocab.pad_id,
        pooling="last",
    )

    ckpt_dir = cfg.artifacts_dir / "checkpoints" / "exp_a_rnn"
    exp_dir = cfg.artifacts_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    rnn_last_res = train_sequence_classifier(
        model=model_last,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        checkpoint_path=ckpt_dir / "best_model.pt",
        id_to_label=id_to_label,
        epochs=epochs,
        lr=lr,
        model_meta={
            "architecture": "VanillaRNNClassifier(pooling='last')",
            "vocab_size": len(vocab),
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        },
    )

    # Also train a mean-pooled Vanilla RNN variant to isolate pooling vs recurrence bottleneck
    set_seed(seed)
    model_mean = VanillaRNNClassifier(
        vocab_size=len(vocab),
        num_classes=len(label_to_id),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.3,
        pad_idx=vocab.pad_id,
        pooling="mean",
    )
    rnn_mean_res = train_sequence_classifier(
        model=model_mean,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        checkpoint_path=ckpt_dir / "best_model_mean_pool.pt",
        id_to_label=id_to_label,
        epochs=epochs,
        lr=lr,
        model_meta={
            "architecture": "VanillaRNNClassifier(pooling='mean')",
            "vocab_size": len(vocab),
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        },
    )

    plot_path = plot_loss_curves(
        {
            "Vanilla RNN (last h_T)": rnn_last_res["history"],
            "Vanilla RNN (mean pool)": rnn_mean_res["history"],
        },
        output_path=cfg.reports_dir / "plots" / "exp_a_rnn_curves.png",
        title="Experiment A: Vanilla RNN on ArXiv Classification",
    )

    payload = {
        "experiment": "Experiment A: Vanilla RNN",
        "dataset": "arxiv_metadata",
        "seed": seed,
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "vocab_size": len(vocab),
        "num_classes": len(label_to_id),
        "classes": list(label_to_id.keys()),
        "baselines": baselines,
        "vanilla_rnn_last": rnn_last_res,
        "vanilla_rnn_mean": rnn_mean_res,
        "loss_curve_plot": str(plot_path),
    }

    out_json = exp_dir / "exp_a_rnn_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved Experiment A results -> %s", out_json)
    return payload


if __name__ == "__main__":
    run_experiment_a()
