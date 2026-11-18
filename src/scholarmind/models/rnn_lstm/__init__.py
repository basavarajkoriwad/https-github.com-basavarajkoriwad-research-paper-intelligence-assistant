"""Custom PyTorch sequential models: Vanilla RNN, LSTM, and BiLSTM + Additive Attention."""

from scholarmind.models.rnn_lstm.dataset import (
    ArxivSequenceDataset,
    collate_padded_sequences,
    compute_classical_baselines,
)
from scholarmind.models.rnn_lstm.lstm_models import (
    AdditiveAttention,
    BiLSTMAttentionClassifier,
    LSTMClassifier,
)
from scholarmind.models.rnn_lstm.trainer import (
    evaluate_sequence_classifier,
    plot_loss_curves,
    train_sequence_classifier,
)
from scholarmind.models.rnn_lstm.vanilla_rnn import VanillaRNNClassifier

__all__ = [
    "AdditiveAttention",
    "ArxivSequenceDataset",
    "BiLSTMAttentionClassifier",
    "LSTMClassifier",
    "VanillaRNNClassifier",
    "collate_padded_sequences",
    "compute_classical_baselines",
    "evaluate_sequence_classifier",
    "plot_loss_curves",
    "train_sequence_classifier",
]
