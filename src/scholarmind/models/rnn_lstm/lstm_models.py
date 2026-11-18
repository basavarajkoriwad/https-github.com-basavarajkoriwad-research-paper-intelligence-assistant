"""Experiment B: Unidirectional LSTM and Bidirectional LSTM + Additive Self-Attention in PyTorch.

Mathematical Foundation:
An LSTM replaces the vanilla RNN's multiplicative state update with an additive memory cell
highway c_t controlled by three sigmoid gates (forget f_t, input i_t, output o_t):

    f_t = sigmoid(W_f [h_{t-1}; x_t] + b_f)        (Forget gate: what to erase from c_{t-1})
    i_t = sigmoid(W_i [h_{t-1}; x_t] + b_i)        (Input gate: what new info to write)
    g_t = tanh(W_g [h_{t-1}; x_t] + b_g)           (Candidate cell update)
    c_t = f_t * c_{t-1} + i_t * g_t                (Additive cell update -> mitigates vanishing grad)
    o_t = sigmoid(W_o [h_{t-1}; x_t] + b_o)        (Output gate: what to expose to h_t)
    h_t = o_t * tanh(c_t)                          (Hidden state)

A Bidirectional LSTM runs a forward LSTM (1 -> T) and backward LSTM (T -> 1), concatenating
their hidden representations at every token:
    h_t = [h_t^(fwd); h_t^(bwd)] in R^{2H}

Additive Self-Attention (Bahdanau-style pooling) computes interpretable token weights alpha_t:
    u_t     = tanh(W_a h_t + b_a)
    alpha_t = softmax_t(v_a^T u_t)   (masked to -inf on <PAD> positions)
    z       = sum_{t=1}^{T} alpha_t h_t
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    """Unidirectional LSTM classifier with explicit access to hidden (h_T) and cell (c_T) states."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling = pooling

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embedded = self.embed_dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, (h_n, c_n) = self.lstm(packed)

        if self.pooling == "mean":
            unpacked, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=input_ids.size(1),
            )
            mask = (
                torch.arange(input_ids.size(1), device=input_ids.device)[None, :]
                < lengths[:, None]
            ).unsqueeze(-1)
            pooled = (unpacked * mask).sum(dim=1) / lengths.clamp(min=1).unsqueeze(-1).float()
        else:
            pooled = h_n[-1]

        logits = self.classifier(self.norm(pooled))
        if return_states:
            return logits, {"h_n": h_n[-1], "c_n": c_n[-1]}
        return logits


class AdditiveAttention(nn.Module):
    """Mask-aware Bahdanau additive self-attention pooling over variable-length sequences."""

    def __init__(self, input_dim: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: (B, T, D)
        energy = torch.tanh(self.proj(hidden_states))  # (B, T, A)
        scores = self.score(energy).squeeze(-1)  # (B, T)

        seq_len = hidden_states.size(1)
        mask = torch.arange(seq_len, device=hidden_states.device)[None, :] < lengths[:, None]
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)  # (B, D)
        return context, weights


class BiLSTMAttentionClassifier(nn.Module):
    """Bidirectional LSTM + Additive Attention returning logits and token-level attention weights."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        attn_dim: int = 128,
        dropout: float = 0.35,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AdditiveAttention(input_dim=hidden_dim * 2, attn_dim=attn_dim)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        return_attention: bool = False,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embedded = self.embed_dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, (h_n, c_n) = self.bilstm(packed)
        unpacked, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=input_ids.size(1),
        )

        context, attn_weights = self.attention(unpacked, lengths)
        logits = self.classifier(self.norm(context))

        if return_states:
            # Concatenate final top-layer forward & backward states
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            c_final = torch.cat([c_n[-2], c_n[-1]], dim=-1)
            return logits, {
                "h_final": h_final,
                "c_final": c_final,
                "attention_weights": attn_weights,
            }
        if return_attention:
            return logits, attn_weights
        return logits
