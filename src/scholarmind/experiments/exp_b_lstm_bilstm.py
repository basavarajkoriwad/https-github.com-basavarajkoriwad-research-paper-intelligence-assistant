"""Experiment B: LSTM and BiLSTM + Additive Attention compared against Vanilla RNN."""

from __future__ import annotations

import json
from functools import partial
from typing import Any

import torch
from torch.utils.data import DataLoader

from scholarmind.core.config import AppConfig, load_config
from scholarmind.core.device import get_device, set_seed
from scholarmind.core.logging import get_logger
from scholarmind.data.storage import load_parquet
from scholarmind.models.rnn_lstm.dataset import (
    ArxivSequenceDataset,
    build_label_mapping,
    collate_padded_sequences,
)
from scholarmind.models.rnn_lstm.lstm_models import (
    BiLSTMAttentionClassifier,
    LSTMClassifier,
)
from scholarmind.models.rnn_lstm.trainer import plot_loss_curves, train_sequence_classifier
from scholarmind.nlp.tokenization import Vocabulary

logger = get_logger(__name__)


@torch.no_grad()
def _inspect_lstm_memory_and_attention(
    lstm_model: LSTMClassifier,
    bilstm_model: BiLSTMAttentionClassifier,
    test_ds: ArxivSequenceDataset,
    id_to_label: dict[int, str],
    device: torch.device,
    sample_idx: int = 0,
) -> dict[str, Any]:
    """Inspect hidden state h_T, memory cell c_T norms, and top-attended tokens on a test paper."""
    lstm_model.eval()
    bilstm_model.eval()

    seq, length, true_label = test_ds[sample_idx]
    tokens = test_ds.raw_tokens[sample_idx]
    input_ids = seq.unsqueeze(0).to(device)
    lengths = torch.tensor([length], dtype=torch.long, device=device)

    _, lstm_states = lstm_model(input_ids, lengths, return_states=True)
    bilstm_logits, bilstm_states = bilstm_model(input_ids, lengths, return_states=True)

    pred_label = id_to_label[int(bilstm_logits.argmax(dim=-1).item())]
    attn = bilstm_states["attention_weights"][0, :length].cpu().tolist()
    top_k = sorted(zip(tokens, attn), key=lambda x: x[1], reverse=True)[:8]

    return {
        "sample_index": sample_idx,
        "true_label": id_to_label[true_label],
        "bilstm_predicted_label": pred_label,
        "sequence_length": length,
        "lstm_hidden_l2_norm": round(float(lstm_states["h_n"].norm().item()), 4),
        "lstm_cell_l2_norm": round(float(lstm_states["c_n"].norm().item()), 4),
        "bilstm_hidden_l2_norm": round(float(bilstm_states["h_final"].norm().item()), 4),
        "bilstm_cell_l2_norm": round(float(bilstm_states["c_final"].norm().item()), 4),
        "top_attended_tokens": [(tok, round(float(w), 4)) for tok, w in top_k],
    }


def run_experiment_b(
    config: AppConfig | None = None,
    epochs: int = 12,
    batch_size: int = 32,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    lr: float = 2e-3,
    max_len: int = 256,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute Experiment B (Unidirectional LSTM & BiLSTM + Additive Attention) on ArXiv."""
    cfg = config or load_config()
    set_seed(seed)
    device = get_device(cfg.hardware.get("preferred_device", "auto"))

    ds_dir = cfg.processed_dir / "arxiv_metadata"
    train_df = load_parquet(ds_dir / "train.parquet")
    val_df = load_parquet(ds_dir / "val.parquet")
    test_df = load_parquet(ds_dir / "test.parquet")
    vocab = Vocabulary.load(ds_dir / "rnn_vocab.json")
    label_to_id, id_to_label = build_label_mapping(train_df)

    train_ds = ArxivSequenceDataset(train_df, vocab, label_to_id, max_len=max_len)
    val_ds = ArxivSequenceDataset(val_df, vocab, label_to_id, max_len=max_len)
    test_ds = ArxivSequenceDataset(test_df, vocab, label_to_id, max_len=max_len)

    collate = partial(collate_padded_sequences, pad_id=vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    ckpt_base = cfg.artifacts_dir / "checkpoints"
    exp_dir = cfg.artifacts_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Unidirectional LSTM
    set_seed(seed)
    lstm_model = LSTMClassifier(
        vocab_size=len(vocab),
        num_classes=len(label_to_id),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        dropout=0.3,
        pad_idx=vocab.pad_id,
        pooling="mean",
    )
    lstm_res = train_sequence_classifier(
        model=lstm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        checkpoint_path=ckpt_base / "exp_b_lstm" / "best_model.pt",
        id_to_label=id_to_label,
        epochs=epochs,
        lr=lr,
        model_meta={"architecture": "LSTMClassifier(unidirectional, mean_pool)"},
    )

    # 2. Bidirectional LSTM + Additive Self-Attention
    set_seed(seed)
    bilstm_model = BiLSTMAttentionClassifier(
        vocab_size=len(vocab),
        num_classes=len(label_to_id),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        attn_dim=128,
        dropout=0.35,
        pad_idx=vocab.pad_id,
    )
    bilstm_res = train_sequence_classifier(
        model=bilstm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        checkpoint_path=ckpt_base / "exp_b_bilstm_attn" / "best_model.pt",
        id_to_label=id_to_label,
        epochs=epochs,
        lr=lr,
        model_meta={"architecture": "BiLSTMAttentionClassifier(2-layer, additive_attention)"},
    )

    state_diagnostics = _inspect_lstm_memory_and_attention(
        lstm_model=lstm_model,
        bilstm_model=bilstm_model,
        test_ds=test_ds,
        id_to_label=id_to_label,
        device=device,
        sample_idx=0,
    )

    plot_path = plot_loss_curves(
        {
            "LSTM (Unidirectional)": lstm_res["history"],
            "BiLSTM + Attention": bilstm_res["history"],
        },
        output_path=cfg.reports_dir / "plots" / "exp_b_lstm_bilstm_curves.png",
        title="Experiment B: LSTM vs. BiLSTM + Attention",
    )

    payload = {
        "experiment": "Experiment B: LSTM and BiLSTM + Attention",
        "dataset": "arxiv_metadata",
        "seed": seed,
        "lstm": lstm_res,
        "bilstm_attention": bilstm_res,
        "state_and_attention_diagnostics": state_diagnostics,
        "loss_curve_plot": str(plot_path),
    }

    out_json = exp_dir / "exp_b_lstm_bilstm_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved Experiment B results -> %s", out_json)
    return payload


if __name__ == "__main__":
    run_experiment_b()
