"""Scientific-aware sentence segmentation protecting academic abbreviations and citations."""

from __future__ import annotations

import re
from typing import Literal

from scholarmind.core.logging import get_logger

logger = get_logger(__name__)

# Common academic abbreviations whose trailing periods must NOT trigger sentence splits
_SCIENTIFIC_ABBREVIATIONS: tuple[str, ...] = (
    "et al.",
    "e.g.",
    "i.e.",
    "vs.",
    "cf.",
    "al.",
    "Fig.",
    "Figs.",
    "Eq.",
    "Eqs.",
    "Sec.",
    "Secs.",
    "Tab.",
    "Ref.",
    "Refs.",
    "No.",
    "Nos.",
    "Vol.",
    "pp.",
    "Dr.",
    "Prof.",
    "Mr.",
    "Ms.",
    "Mrs.",
    "Inc.",
    "Ltd.",
    "approx.",
    "resp.",
    "w.r.t.",
    "s.t.",
)

_DOT_SENTINEL = "<DOT>"

# Matches decimals / version numbers like 3.14, 0.95, v2.0, Sec. 3.2.1
_DECIMAL_PATTERN = re.compile(r"(?<=\d)\.(?=\d)")

# Fallback regex sentence boundary splitter (punctuation + space + capital/bracket/quote/math)
_FALLBACK_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\(\"'\$\[])")


class ScientificSentenceSegmenter:
    """Sentence segmenter designed for research papers (supports NLTK, spaCy, and regex fallback).

    Protects academic abbreviations (`et al.`, `Fig.`, `Eq.`, `e.g.`, `w.r.t.`) and
    numeric decimals (`3.14`, `Sec. 3.2`) before delegating to the underlying segmenter.
    """

    def __init__(
        self,
        backend: Literal["auto", "spacy", "nltk", "regex"] = "auto",
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        self.requested_backend = backend
        self.spacy_model = spacy_model
        self._spacy_nlp = None
        self._nltk_ready = False
        self.active_backend = self._initialize_backend(backend)

    def _initialize_backend(self, backend: str) -> str:
        if backend in ("auto", "spacy"):
            try:
                import spacy

                try:
                    self._spacy_nlp = spacy.load(
                        self.spacy_model,
                        disable=["ner", "lemmatizer", "tagger", "attribute_ruler"],
                    )
                    if "parser" not in self._spacy_nlp.pipe_names and "sentencizer" not in self._spacy_nlp.pipe_names:
                        self._spacy_nlp.add_pipe("sentencizer")
                except OSError:
                    # Fallback cleanly to lightweight blank English pipeline + sentencizer
                    self._spacy_nlp = spacy.blank("en")
                    self._spacy_nlp.add_pipe("sentencizer")
                return "spacy"
            except ImportError:
                if backend == "spacy":
                    raise

        if backend in ("auto", "nltk"):
            try:
                import nltk

                try:
                    nltk.data.find("tokenizers/punkt_tab")
                    self._nltk_ready = True
                    return "nltk"
                except LookupError:
                    try:
                        nltk.data.find("tokenizers/punkt")
                        self._nltk_ready = True
                        return "nltk"
                    except LookupError:
                        logger.warning(
                            "NLTK punkt tokenizer data not found locally; falling back to regex splitter."
                        )
            except ImportError:
                if backend == "nltk":
                    raise

        return "regex"

    @staticmethod
    def _protect_scientific_periods(text: str) -> str:
        protected = _DECIMAL_PATTERN.sub(_DOT_SENTINEL, text)
        for abbr in _SCIENTIFIC_ABBREVIATIONS:
            # Protect exact abbreviation case-insensitively at word boundary
            pattern = re.compile(r"(?<!\w)" + re.escape(abbr), re.IGNORECASE)
            protected = pattern.sub(lambda m: m.group(0).replace(".", _DOT_SENTINEL), protected)
        return protected

    @staticmethod
    def _restore_periods(sentences: list[str]) -> list[str]:
        return [s.replace(_DOT_SENTINEL, ".").strip() for s in sentences if s.strip()]

    def segment(self, text: str | None) -> list[str]:
        """Split scientific prose into clean sentences."""
        if not text or not isinstance(text, str):
            return []

        stripped = text.strip()
        if not stripped:
            return []

        protected = self._protect_scientific_periods(stripped)

        if self.active_backend == "spacy" and self._spacy_nlp is not None:
            # Handle very long documents safely without exceeding spaCy default max_length
            if len(protected) > self._spacy_nlp.max_length:
                self._spacy_nlp.max_length = len(protected) + 1000
            doc = self._spacy_nlp(protected)
            raw_sents = [sent.text for sent in doc.sents]
        elif self.active_backend == "nltk" and self._nltk_ready:
            import nltk

            raw_sents = nltk.sent_tokenize(protected)
        else:
            raw_sents = _FALLBACK_SPLIT_PATTERN.split(protected)

        return self._restore_periods(raw_sents)

    def segment_batch(self, texts: list[str | None]) -> list[list[str]]:
        """Segment a batch of texts into lists of sentences."""
        return [self.segment(t) for t in texts]
