"""Experiment C: Pretrained BERT / SciBERT contextual representations and fine-tuning in PyTorch.

Mathematical Foundation:
BERT is a multi-layer bidirectional Transformer encoder built on Scaled Dot-Product
Multi-Head Self-Attention:

    Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V
    MultiHead(H)       = Concat(head_1, ..., head_h) W^O,
                         where head_i = Attention(H W_i^Q, H W_i^K, H W_i^V)

Unlike static word embeddings (where a word has one fixed vector regardless of context),
each BERT token representation h_i^(L) integrates bidirectional context from ALL tokens
simultaneously. Attention-masked mean pooling produces a document embedding:

    e_doc = (sum_{i=1}^{T} m_i * h_i^(L)) / (sum_{i=1}^{T} m_i),   m_i in {0, 1}

Fine-tuning adds a linear classification head z = W_cls Dropout(e_doc) + b_cls and optimizes
cross-entropy loss end-to-end (or over the top un-frozen Transformer layers for CPU efficiency).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from scholarmind.core.device import count_parameters
from scholarmind.core.logging import get_logger

logger = get_logger(__name__)


class ArxivBERTDataset(Dataset):
    """Tokenizes cleaned scientific papers for BERT / SciBERT fine-tuning."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        label_to_id: dict[str, int],
        text_col: str = "combined_text",
        label_col: str = "primary_category",
        max_length: int = 192,
    ) -> None:
        self.texts = df[text_col].astype(str).tolist()
        self.labels = [label_to_id[x] for x in df[label_col].astype(str)]
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class SciBERTClassifier(nn.Module):
    """BERT / SciBERT encoder with selective layer freezing and mean/[CLS] pooling."""

    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        num_classes: int = 8,
        dropout: float = 0.25,
        pooling: str = "mean",
        unfreeze_last_n_layers: int = 2,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pooling = pooling
        self.unfreeze_last_n_layers = unfreeze_last_n_layers

        self.encoder = backbone if backbone is not None else AutoModel.from_pretrained(model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        if unfreeze_last_n_layers >= 0:
            self._configure_trainable_layers(unfreeze_last_n_layers)

    def _configure_trainable_layers(self, unfreeze_last_n: int) -> None:
        """Freeze lower Transformer layers for practical CPU fine-tuning while adapting top layers."""
        for param in self.encoder.parameters():
            param.requires_grad = False

        encoder_layers = None
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            encoder_layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, "transformer") and hasattr(self.encoder.transformer, "layer"):
            encoder_layers = self.encoder.transformer.layer

        if encoder_layers is not None and unfreeze_last_n > 0:
            for layer in encoder_layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

        if hasattr(self.encoder, "pooler") and self.encoder.pooler is not None:
            for param in self.encoder.pooler.parameters():
                param.requires_grad = True

    def pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool token representations into a single fixed-dimensional scientific document vector."""
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def extract_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Extract contextual document embeddings from the BERT backbone."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(outputs.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=-1) if normalize else pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(outputs.last_hidden_state, attention_mask)
        return self.classifier(self.dropout(self.norm(pooled)))


@torch.no_grad()
def demonstrate_contextual_embeddings(
    model_name: str,
    device: torch.device,
    tokenizer: Any | None = None,
    encoder: nn.Module | None = None,
) -> dict[str, Any]:
    """Empirical demonstration that BERT generates context-dependent token embeddings.

    Computes pairwise cosine similarity of the exact same target scientific word (`kernel`
    and `attention`) across distinct scientific disciplines vs. similar scientific contexts.
    """
    tok = tokenizer or AutoTokenizer.from_pretrained(model_name)
    enc = (encoder or AutoModel.from_pretrained(model_name)).to(device)
    enc.eval()

    probes = [
        (
            "kernel",
            "Support vector machines use a radial basis kernel function for nonlinear classification.",
            "Gaussian processes define a covariance kernel over continuous function spaces.",
            "The operating system kernel schedules threads across physical processor cores.",
        ),
        (
            "attention",
            "Multi-head self-attention weights token interactions inside Transformer encoders.",
            "Cross-attention aligns decoder queries with encoder keys during summarization.",
            "Visual attention in cognitive neuroscience guides human eye fixation on salient stimuli.",
        ),
    ]

    results: list[dict[str, Any]] = []
    for target_word, sent_a, sent_b_similar, sent_c_different in probes:
        vecs: list[torch.Tensor] = []
        for sentence in (sent_a, sent_b_similar, sent_c_different):
            encoded = tok(sentence, return_tensors="pt").to(device)
            tokens = tok.convert_ids_to_tokens(encoded["input_ids"][0])
            out = enc(**encoded).last_hidden_state[0]
            # Locate target word subword span
            matched_indices = [
                i for i, t in enumerate(tokens) if target_word in t.replace("##", "").lower()
            ]
            if matched_indices:
                vec = out[matched_indices].mean(dim=0)
            else:
                vec = out[1:-1].mean(dim=0)
            vecs.append(F.normalize(vec, p=2, dim=-1).cpu())

        sim_same_domain = float(torch.dot(vecs[0], vecs[1]).item())
        sim_cross_domain = float(torch.dot(vecs[0], vecs[2]).item())
        results.append(
            {
                "target_word": target_word,
                "same_domain_cosine_sim": round(sim_same_domain, 4),
                "cross_domain_cosine_sim": round(sim_cross_domain, 4),
                "contextual_shift_gap": round(sim_same_domain - sim_cross_domain, 4),
            }
        )

    return {
        "model_name": model_name,
        "probes": results,
    }


@torch.no_grad()
def evaluate_bert_classifier(
    model: SciBERTClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate BERT/SciBERT classifier on a DataLoader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    t0 = time.perf_counter()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_targets.extend(labels.cpu().tolist())
    elapsed = time.perf_counter() - t0

    return {
        "loss": round(total_loss / max(1, total_samples), 4),
        "accuracy": round(float(accuracy_score(all_targets, all_preds)), 4),
        "macro_f1": round(float(f1_score(all_targets, all_preds, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(all_targets, all_preds, average="weighted", zero_division=0)), 4),
        "inference_ms_per_sample": round((elapsed / max(1, total_samples)) * 1000.0, 3),
        "predictions": all_preds,
        "targets": all_targets,
    }


def train_bert_classifier(
    model: SciBERTClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    checkpoint_path: Path,
    id_to_label: dict[int, str],
    epochs: int = 4,
    lr: float = 5e-5,
    head_lr: float = 5e-4,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
) -> dict[str, Any]:
    """Fine-tune BERT/SciBERT with differential learning rates (backbone vs. classification head)."""
    model = model.to(device)
    total_params, trainable_params = count_parameters(model)
    criterion = nn.CrossEntropyLoss()

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [
        p
        for n, p in model.named_parameters()
        if not n.startswith("encoder.") and p.requires_grad
    ]
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            bs = labels.size(0)
            running_loss += float(loss.item()) * bs
            seen += bs

        scheduler.step()
        train_loss = running_loss / max(1, seen)
        val_metrics = evaluate_bert_classifier(model, val_loader, criterion, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_record)
        logger.info(
            "BERT Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_f1=%.4f",
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
            torch.save(
                {
                    "model_name": model.model_name,
                    "pooling": model.pooling,
                    "unfreeze_last_n_layers": model.unfreeze_last_n_layers,
                    "model_state_dict": model.state_dict(),
                    "id_to_label": id_to_label,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "history": history,
                },
                checkpoint_path,
            )

    train_seconds = round(time.perf_counter() - train_start, 2)

    best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    best_val = evaluate_bert_classifier(model, val_loader, criterion, device)
    test_metrics = evaluate_bert_classifier(model, test_loader, criterion, device)

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
        "model_name": model.model_name,
        "device": str(device),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "unfreeze_last_n_layers": model.unfreeze_last_n_layers,
        "train_seconds": train_seconds,
        "best_epoch": best_epoch,
        "checkpoint_path": str(checkpoint_path),
        "history": history,
        "best_val": {k: v for k, v in best_val.items() if k not in ("predictions", "targets")},
        "test": {k: v for k, v in test_metrics.items() if k not in ("predictions", "targets")},
        "per_class_test_report": cls_rep,
    }
