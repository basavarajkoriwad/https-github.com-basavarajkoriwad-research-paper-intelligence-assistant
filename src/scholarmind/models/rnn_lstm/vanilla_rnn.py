"""Experiment A: Vanilla RNN sequence classifier implemented in PyTorch.

Mathematical Foundation:
Given an embedded token sequence x_1, ..., x_T (x_t in R^d), a basic Elman RNN updates its
recurrent hidden state at each timestep via:

    h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh),   h_0 = 0

Backpropagation Through Time (BPTT) propagates gradients across T timesteps:
    dL / dh_t = (dL / dh_T) * prod_{k=t+1}^{T} diag(1 - h_k^2) W_hh

Because |1 - h_k^2| <= 1 and repeated multiplications by W_hh shrink exponentially when
spectral_radius(W_hh) < 1 (or explode when > 1), vanilla RNNs struggle to preserve
long-range dependencies across 200-word scientific abstracts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VanillaRNNClassifier(nn.Module):
    """Basic multi-layer Elman RNN classifier with length-aware packing."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0,
        pooling: str = "last",
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
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run packed RNN recurrence and return class logits."""
        embedded = self.embed_dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, h_n = self.rnn(packed)

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
            # Final timestep hidden state of the top RNN layer: shape (B, hidden_dim)
            pooled = h_n[-1]

        logits = self.classifier(pooled)
        if return_hidden:
            return logits, h_n[-1]
        return logits
