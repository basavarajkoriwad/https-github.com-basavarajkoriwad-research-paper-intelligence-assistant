"""Task-aware text cleaning and normalization for scientific literature and QA corpora."""

from __future__ import annotations

import html
import re
import unicodedata
from enum import Enum
from typing import Any


class CleaningProfile(str, Enum):
    """Task-specific cleaning profiles balancing fidelity and noise reduction."""

    QA_PRESERVE_OFFSETS = "qa_preserve_offsets"
    SUMMARIZATION_SCIENTIFIC = "summarization_scientific"
    CLASSIFICATION_CLASSICAL = "classification_classical"
    RAG_SEMANTIC = "rag_semantic"


_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u00a0": " ",
    "\u200b": "",
    "\ufeff": "",
}

_LIGATURE_PATTERN = re.compile("|".join(re.escape(k) for k in _LIGATURE_MAP))

# Matches words broken across PDF line wraps: "trans-\nformer" -> "transformer"
_DEHYPHEN_PATTERN = re.compile(r"([A-Za-z]{2,})-\s*\n\s*([a-z]{2,})")

# Control characters except \n and \t
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# ccdv/arxiv-summarization special tokens
_XMATH_PATTERN = re.compile(r"@xmath\d+", re.IGNORECASE)
_XCITE_PATTERN = re.compile(r"@xcite", re.IGNORECASE)
_XREF_PATTERN = re.compile(r"@xref", re.IGNORECASE)

# Standard LaTeX math blocks & inline expressions
_LATEX_BLOCK_PATTERN = re.compile(
    r"(\$\$[\s\S]*?\$\$|\\begin\{(?:equation|align|eqnarray|gather)\*?\}[\s\S]*?\\end\{(?:equation|align|eqnarray|gather)\*?\})",
    re.MULTILINE,
)
_LATEX_INLINE_PATTERN = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$")

# Citation markers like [1], [1, 2], [12-15]
_NUMERIC_CITATION_PATTERN = re.compile(r"\[\s*\d+(?:\s*[-,–]\s*\d+)*\s*\]")
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_MULTI_SPACE_PATTERN = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")


class ScientificTextCleaner:
    """Configurable text cleaner tailored to scientific papers and QA datasets.

    Design rationale:
    - Preserves scientific punctuation, casing, and equations for Transformer/RAG tasks.
    - Normalizes PDF ligatures, broken hyphenations, and ccdv/arxiv-summarization markers
      (`@xmath0`, `@xcite`, `@xref`).
    - Supports aggressive normalization (lowercasing, URL/citation stripping) strictly for
      classical NLP / RNN-LSTM vocabulary pipelines where sparsity hurts generalization.
    """

    def __init__(
        self,
        profile: CleaningProfile = CleaningProfile.RAG_SEMANTIC,
        normalize_arxiv_placeholders: bool = True,
        replace_latex_with_token: bool = False,
        strip_urls: bool = False,
        strip_numeric_citations: bool = False,
        lowercase: bool = False,
        preserve_paragraphs: bool = True,
    ) -> None:
        self.profile = profile
        self.normalize_arxiv_placeholders = normalize_arxiv_placeholders
        self.replace_latex_with_token = replace_latex_with_token
        self.strip_urls = strip_urls
        self.strip_numeric_citations = strip_numeric_citations
        self.lowercase = lowercase
        self.preserve_paragraphs = preserve_paragraphs

        if profile == CleaningProfile.CLASSIFICATION_CLASSICAL:
            self.replace_latex_with_token = True
            self.strip_urls = True
            self.strip_numeric_citations = True
            self.lowercase = True
            self.preserve_paragraphs = False
        elif profile == CleaningProfile.SUMMARIZATION_SCIENTIFIC:
            self.normalize_arxiv_placeholders = True
            self.replace_latex_with_token = False
            self.strip_urls = True
            self.strip_numeric_citations = False
            self.lowercase = False
            self.preserve_paragraphs = True
        elif profile == CleaningProfile.RAG_SEMANTIC:
            self.normalize_arxiv_placeholders = True
            self.replace_latex_with_token = False
            self.strip_urls = False
            self.strip_numeric_citations = False
            self.lowercase = False
            self.preserve_paragraphs = True
        elif profile == CleaningProfile.QA_PRESERVE_OFFSETS:
            self.normalize_arxiv_placeholders = False
            self.replace_latex_with_token = False
            self.strip_urls = False
            self.strip_numeric_citations = False
            self.lowercase = False
            self.preserve_paragraphs = False

    @staticmethod
    def has_math(text: str) -> bool:
        """Detect whether text contains LaTeX math or arXiv math placeholders."""
        if not text:
            return False
        return bool(
            _LATEX_BLOCK_PATTERN.search(text)
            or _LATEX_INLINE_PATTERN.search(text)
            or _XMATH_PATTERN.search(text)
            or "[MATH]" in text
        )

    @staticmethod
    def has_citation(text: str) -> bool:
        """Detect whether text contains numeric, author-year, or arXiv citation markers."""
        if not text:
            return False
        return bool(
            _NUMERIC_CITATION_PATTERN.search(text)
            or _XCITE_PATTERN.search(text)
            or "[CITE]" in text
            or re.search(r"\bet\s+al\.", text)
        )

    def clean(self, text: str | None) -> str:
        """Clean and normalize a single text string according to the active profile."""
        if not text or not isinstance(text, str):
            return ""

        cleaned = html.unescape(text)
        cleaned = _LIGATURE_PATTERN.sub(lambda m: _LIGATURE_MAP[m.group(0)], cleaned)
        cleaned = unicodedata.normalize("NFKC", cleaned)
        cleaned = _CONTROL_CHAR_PATTERN.sub("", cleaned)

        # Normalize line endings
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # Repair PDF hyphenation across line breaks
        cleaned = _DEHYPHEN_PATTERN.sub(r"\1\2", cleaned)

        # Normalize ccdv/arxiv-summarization special markers
        if self.normalize_arxiv_placeholders:
            cleaned = _XMATH_PATTERN.sub("[MATH]", cleaned)
            cleaned = _XCITE_PATTERN.sub("[CITE]", cleaned)
            cleaned = _XREF_PATTERN.sub("[REF]", cleaned)

        if self.replace_latex_with_token:
            cleaned = _LATEX_BLOCK_PATTERN.sub(" [MATH] ", cleaned)
            cleaned = _LATEX_INLINE_PATTERN.sub(" [MATH] ", cleaned)

        if self.strip_urls:
            cleaned = _URL_PATTERN.sub(" ", cleaned)

        if self.strip_numeric_citations:
            cleaned = _NUMERIC_CITATION_PATTERN.sub(" ", cleaned)

        # Collapse whitespace while optionally preserving paragraph structure
        if self.preserve_paragraphs:
            lines = [_MULTI_SPACE_PATTERN.sub(" ", line).strip() for line in cleaned.split("\n")]
            cleaned = "\n".join(lines)
            cleaned = _MULTI_NEWLINE_PATTERN.sub("\n\n", cleaned)
        else:
            cleaned = re.sub(r"\s+", " ", cleaned)

        if self.lowercase:
            cleaned = cleaned.lower()

        return cleaned.strip()

    def clean_batch(self, texts: list[str | None]) -> list[str]:
        """Clean a list of texts."""
        return [self.clean(t) for t in texts]


def _normalize_qa_string(text: str) -> str:
    """Minimal safe normalization for QA strings prior to offset alignment."""
    text = _LIGATURE_PATTERN.sub(lambda m: _LIGATURE_MAP[m.group(0)], text)
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHAR_PATTERN.sub("", text)
    return text


def clean_qa_example(
    context: str,
    question: str,
    answer_text: str | None = None,
    answer_start: int | None = None,
) -> dict[str, Any]:
    """Offset-safe cleaner for extractive QA (e.g., SQuAD 2.0).

    If normalization shifts character positions (e.g., ligature expansion or whitespace
    trimming), this function realigns `answer_start` near the original index and verifies
    exact span match `cleaned_context[new_start:new_start+len(cleaned_answer)] == cleaned_answer`.
    """
    cleaned_question = re.sub(r"\s+", " ", _normalize_qa_string(question or "")).strip()
    raw_context = context or ""
    cleaned_context = _normalize_qa_string(raw_context)

    if answer_text is None or answer_text == "" or answer_start is None or answer_start < 0:
        return {
            "context": cleaned_context.strip(),
            "question": cleaned_question,
            "answer_text": "",
            "answer_start": -1,
            "offset_valid": True,
        }

    cleaned_answer = _normalize_qa_string(answer_text).strip()
    leading_ws = len(cleaned_context) - len(cleaned_context.lstrip())
    stripped_context = cleaned_context.strip()

    # Estimate adjusted start position and search locally around it first
    approx_start = max(0, int(answer_start) - leading_ws)
    window_start = max(0, approx_start - 32)
    window_end = min(len(stripped_context), approx_start + len(cleaned_answer) + 64)

    rel_pos = stripped_context.find(cleaned_answer, window_start, window_end)
    if rel_pos == -1:
        rel_pos = stripped_context.find(cleaned_answer)

    offset_valid = (
        rel_pos != -1
        and stripped_context[rel_pos : rel_pos + len(cleaned_answer)] == cleaned_answer
    )

    return {
        "context": stripped_context,
        "question": cleaned_question,
        "answer_text": cleaned_answer if offset_valid else "",
        "answer_start": rel_pos if offset_valid else -1,
        "offset_valid": offset_valid,
    }
