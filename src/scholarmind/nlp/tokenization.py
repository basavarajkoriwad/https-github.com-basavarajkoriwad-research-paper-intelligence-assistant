"""Multi-backend scientific tokenization and PyTorch-ready vocabulary management."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Literal

# Preserves special placeholders ([MATH], [CITE]), hyphenated technical terms (self-attention,
# DeBERTa-v3, F1-score), decimals (92.4, 1e-4), contractions, words, and punctuation.
_SCIENTIFIC_TOKEN_PATTERN = re.compile(
    r"\[(?:MATH|CITE|REF)\]"
    r"|[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)+"
    r"|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    r"|[A-Za-z][A-Za-z0-9]*"
    r"|[^\sA-Za-z0-9]"
)


class ScientificTokenizer:
    """Tokenizer supporting scientific regex, NLTK, and spaCy backends."""

    def __init__(
        self,
        backend: Literal["scientific_regex", "nltk", "spacy"] = "scientific_regex",
        lowercase: bool = False,
        strip_punctuation: bool = False,
    ) -> None:
        self.backend = backend
        self.lowercase = lowercase
        self.strip_punctuation = strip_punctuation
        self._spacy_nlp = None

        if backend == "spacy":
            import spacy

            self._spacy_nlp = spacy.blank("en")

    def tokenize(self, text: str | None) -> list[str]:
        """Tokenize a text string into tokens."""
        if not text or not isinstance(text, str):
            return []

        source = text.lower() if self.lowercase else text

        if self.backend == "spacy" and self._spacy_nlp is not None:
            tokens = [tok.text for tok in self._spacy_nlp.tokenizer(source) if not tok.is_space]
        elif self.backend == "nltk":
            from nltk.tokenize import wordpunct_tokenize

            tokens = wordpunct_tokenize(source)
        else:
            tokens = _SCIENTIFIC_TOKEN_PATTERN.findall(source)

        if self.strip_punctuation:
            tokens = [
                tok
                for tok in tokens
                if re.search(r"[A-Za-z0-9]", tok) or tok in ("[MATH]", "[CITE]", "[REF]")
            ]

        return tokens

    def tokenize_batch(self, texts: Iterable[str | None]) -> list[list[str]]:
        """Tokenize an iterable of texts."""
        return [self.tokenize(t) for t in texts]


class Vocabulary:
    """Frequency-filtered vocabulary with special tokens for PyTorch RNN/BiLSTM models."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(
        self,
        min_freq: int = 2,
        max_size: int | None = 30000,
    ) -> None:
        self.min_freq = min_freq
        self.max_size = max_size
        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]
        self.token_to_id: dict[str, int] = {
            tok: idx for idx, tok in enumerate(self.special_tokens)
        }
        self.id_to_token: list[str] = list(self.special_tokens)
        self.frequencies: Counter[str] = Counter()

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def fit(self, tokenized_corpus: Iterable[list[str]]) -> "Vocabulary":
        """Build vocabulary strictly from TRAINING split tokens (never val/test!)."""
        self.frequencies.clear()
        for tokens in tokenized_corpus:
            self.frequencies.update(tokens)

        eligible = [
            (tok, freq)
            for tok, freq in self.frequencies.most_common()
            if freq >= self.min_freq and tok not in self.token_to_id
        ]
        if self.max_size is not None:
            capacity = max(0, self.max_size - len(self.special_tokens))
            eligible = eligible[:capacity]

        self.token_to_id = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.id_to_token = list(self.special_tokens)

        for tok, _ in eligible:
            self.token_to_id[tok] = len(self.id_to_token)
            self.id_to_token.append(tok)

        return self

    def encode(
        self,
        tokens: list[str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: int | None = None,
    ) -> list[int]:
        """Convert tokens into integer IDs."""
        ids: list[int] = []
        if add_bos:
            ids.append(self.token_to_id[self.BOS_TOKEN])
        ids.extend(self.token_to_id.get(tok, self.unk_id) for tok in tokens)
        if add_eos:
            ids.append(self.token_to_id[self.EOS_TOKEN])
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> list[str]:
        """Convert integer IDs back to tokens."""
        special_set = set(self.special_tokens)
        result: list[str] = []
        for idx in ids:
            tok = self.id_to_token[idx] if 0 <= idx < len(self.id_to_token) else self.UNK_TOKEN
            if skip_special and tok in special_set:
                continue
            result.append(tok)
        return result

    def save(self, path: Path) -> None:
        """Persist vocabulary to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "min_freq": self.min_freq,
            "max_size": self.max_size,
            "id_to_token": self.id_to_token,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """Load vocabulary from JSON."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        vocab = cls(min_freq=payload["min_freq"], max_size=payload["max_size"])
        vocab.id_to_token = list(payload["id_to_token"])
        vocab.token_to_id = {tok: idx for idx, tok in enumerate(vocab.id_to_token)}
        return vocab


def estimate_subword_tokens(text: str | None) -> int:
    """Estimate Transformer subword token count for scientific prose.

    Scientific English averages ~1.30 WordPiece/BPE subword tokens per whitespace word
    due to technical terms, symbols, and hyphenations. Using this lightweight calibrated
    estimator enables fast CPU chunking and EDA without loading large tokenizer weights.
    """
    if not text or not isinstance(text, str):
        return 0
    words = text.split()
    if not words:
        return 0
    return max(1, int(round(len(words) * 1.30)))
