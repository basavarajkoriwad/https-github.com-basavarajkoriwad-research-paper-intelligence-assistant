"""Reusable PyTorch training, checkpointing, evaluation, and curve plotting for sequence models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scholarmind.core.device import count_parameters
from scholarmind.core.logging import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def evaluate_sequence_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a PyTorch sequence classifier on a validation or test DataLoader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    t0 = time.perf_counter()
    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().tolist())
    elapsed = time.perf_counter() - t0

    avg_loss = total_loss / max(1, total_samples)
    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))
    latency_ms = (elapsed / max(1, total_samples)) * 1000.0

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "inference_ms_per_sample": round(latency_ms, 3),
        "predictions": all_preds,
        "targets": all_targets,
    }


def train_sequence_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    checkpoint_path: Path,
    id_to_label: dict[int, str],
    epochs: int = 12,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    patience: int = 4,
    model_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a sequence classifier, record per-epoch losses, save best checkpoint, and test."""
    model = model.to(device)
    total_params, trainable_params = count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        grad_norms: list[float] = []

        for input_ids, lengths, labels in train_loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()

            total_norm = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip))
            grad_norms.append(total_norm)
            optimizer.step()

            bs = labels.size(0)
            running_loss += float(loss.item()) * bs
            seen += bs

        scheduler.step()
        train_loss = running_loss / max(1, seen)
        val_metrics = evaluate_sequence_classifier(model, val_loader, criterion, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "avg_grad_norm": round(float(np.mean(grad_norms)), 4) if grad_norms else 0.0,
            "lr": round(float(optimizer.param_groups[0]["lr"]), 6),
        }
        history.append(epoch_record)
        logger.info(
            "Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_f1=%.4f",
            epoch,
            epochs,
            epoch_record["train_loss"],
            epoch_record["val_loss"],
            epoch_record["val_accuracy"],
            epoch_record["val_macro_f1"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "id_to_label": id_to_label,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "model_meta": model_meta or {},
                },
                checkpoint_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered at epoch %d (best epoch=%d).", epoch, best_epoch)
                break

    train_seconds = round(time.perf_counter() - train_start, 2)

    # Reload best checkpoint before held-out test evaluation
    best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    best_val = evaluate_sequence_classifier(model, val_loader, criterion, device)
    test_metrics = evaluate_sequence_classifier(model, test_loader, criterion, device)

    target_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
    cls_rep = classification_report(
        test_metrics["targets"],
        test_metrics["predictions"],
        labels=sorted(id_to_label.keys()),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "device": str(device),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "train_seconds": train_seconds,
        "best_epoch": best_epoch,
        "checkpoint_path": str(checkpoint_path),
        "history": history,
        "best_val": {
            k: v for k, v in best_val.items() if k not in ("predictions", "targets")
        },
        "test": {
            k: v for k, v in test_metrics.items() if k not in ("predictions", "targets")
        },
        "per_class_test_report": cls_rep,
    }


def plot_loss_curves(
    series_map: dict[str, list[dict[str, float]]],
    output_path: Path,
    title: str = "Training & Validation Loss Curves",
) -> Path:
    """Plot train/val loss curves for one or more models side-by-side."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, hist in series_map.items():
        epochs = [h["epoch"] for h in hist]
        tr_loss = [h["train_loss"] for h in hist]
        va_loss = [h["val_loss"] for h in hist]
        axes[0].plot(epochs, tr_loss, marker="o", label=f"{name} (train)")
        axes[0].plot(epochs, va_loss, marker="s", linestyle="--", label=f"{name} (val)")

        if "val_macro_f1" in hist[0]:
            va_f1 = [h["val_macro_f1"] for h in hist]
            axes[1].plot(epochs, va_f1, marker="o", label=f"{name} (val Macro-F1)")

    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title(f"{title} — Validation Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
