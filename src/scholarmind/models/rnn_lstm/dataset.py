"""PyTorch Dataset, dynamic padding collate function, and classical baselines for ArXiv text."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset

from scholarmind.nlp.cleaning import CleaningProfile, ScientificTextCleaner
from scholarmind.nlp.tokenization import ScientificTokenizer, Vocabulary


class ArxivSequenceDataset(Dataset):
    """Encodes scientific paper titles + abstracts into variable-length integer token sequences."""

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        label_to_id: dict[str, int],
        text_col: str = "combined_text",
        label_col: str = "primary_category",
        max_len: int = 256,
    ) -> None:
        self.vocab = vocab
        self.label_to_id = label_to_id
        self.max_len = max_len
        self.cleaner = ScientificTextCleaner(profile=CleaningProfile.CLASSIFICATION_CLASSICAL)
        self.tokenizer = ScientificTokenizer(
            backend="scientific_regex",
            lowercase=True,
            strip_punctuation=True,
        )

        self.sequences: list[list[int]] = []
        self.labels: list[int] = []
        self.raw_tokens: list[list[str]] = []

        for text, label in zip(df[text_col].astype(str), df[label_col].astype(str)):
            cleaned = self.cleaner.clean(text)
            tokens = self.tokenizer.tokenize(cleaned)[: self.max_len]
            encoded = self.vocab.encode(tokens, max_len=self.max_len)
            if not encoded:
                encoded = [self.vocab.unk_id]
                tokens = [Vocabulary.UNK_TOKEN]
            self.sequences.append(encoded)
            self.raw_tokens.append(tokens)
            self.labels.append(self.label_to_id[label])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        return seq, len(self.sequences[idx]), self.labels[idx]


def collate_padded_sequences(
    batch: list[tuple[torch.Tensor, int, int]],
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamically pad sequences to the longest example in the current mini-batch."""
    seqs, lengths, labels = zip(*batch)
    max_batch_len = max(lengths)
    padded = torch.full((len(batch), max_batch_len), fill_value=pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, : len(seq)] = seq
    return (
        padded,
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def build_label_mapping(train_df: pd.DataFrame, label_col: str = "primary_category") -> tuple[dict[str, int], dict[int, str]]:
    """Create deterministic sorted label <-> integer ID maps strictly from training labels."""
    classes = sorted(train_df[label_col].astype(str).unique().tolist())
    label_to_id = {lbl: idx for idx, lbl in enumerate(classes)}
    id_to_label = {idx: lbl for lbl, idx in label_to_id.items()}
    return label_to_id, id_to_label


def compute_classical_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "combined_text",
    label_col: str = "primary_category",
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Compute Majority-Class and TF-IDF + Logistic Regression baselines on identical splits."""
    label_to_id, _ = build_label_mapping(train_df, label_col=label_col)
    y_train = np.array([label_to_id[x] for x in train_df[label_col].astype(str)])
    y_val = np.array([label_to_id[x] for x in val_df[label_col].astype(str)])
    y_test = np.array([label_to_id[x] for x in test_df[label_col].astype(str)])

    # 1. Majority-class baseline
    majority_class = int(np.bincount(y_train).argmax())
    maj_val_pred = np.full_like(y_val, majority_class)
    maj_test_pred = np.full_like(y_test, majority_class)

    # 2. TF-IDF + L2-regularized Logistic Regression baseline (fit strictly on train!)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000,
        sublinear_tf=True,
        min_df=2,
    )
    x_train = vectorizer.fit_transform(train_df[text_col].astype(str))
    x_val = vectorizer.transform(val_df[text_col].astype(str))
    x_test = vectorizer.transform(test_df[text_col].astype(str))

    clf = LogisticRegression(max_iter=1000, random_state=seed, C=2.0)
    clf.fit(x_train, y_train)
    lr_val_pred = clf.predict(x_val)
    lr_test_pred = clf.predict(x_test)

    return {
        "majority_baseline": {
            "val_accuracy": round(float(accuracy_score(y_val, maj_val_pred)), 4),
            "val_macro_f1": round(float(f1_score(y_val, maj_val_pred, average="macro", zero_division=0)), 4),
            "test_accuracy": round(float(accuracy_score(y_test, maj_test_pred)), 4),
            "test_macro_f1": round(float(f1_score(y_test, maj_test_pred, average="macro", zero_division=0)), 4),
        },
        "tfidf_logreg_baseline": {
            "val_accuracy": round(float(accuracy_score(y_val, lr_val_pred)), 4),
            "val_macro_f1": round(float(f1_score(y_val, lr_val_pred, average="macro", zero_division=0)), 4),
            "test_accuracy": round(float(accuracy_score(y_test, lr_test_pred)), 4),
            "test_macro_f1": round(float(f1_score(y_test, lr_test_pred, average="macro", zero_division=0)), 4),
            "vocab_features": int(len(vectorizer.vocabulary_)),
        },
    }
