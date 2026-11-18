"""Unit tests for scientific text cleaning and offset-safe QA normalization."""

from __future__ import annotations

from scholarmind.nlp.cleaning import (
    CleaningProfile,
    ScientificTextCleaner,
    clean_qa_example,
)


def test_ligatures_dehyphenation_and_arxiv_placeholders() -> None:
    cleaner = ScientificTextCleaner(profile=CleaningProfile.SUMMARIZATION_SCIENTIFIC)
    raw = (
        "We propose a ﬁltered trans-\nformer architecture @xcite achieving @xmath0 gain.\n"
        "See @xref and https://example.org/repo for details."
    )
    cleaned = cleaner.clean(raw)

    assert "filtered" in cleaned
    assert "transformer" in cleaned
    assert "[CITE]" in cleaned
    assert "[MATH]" in cleaned
    assert "[REF]" in cleaned
    assert "https://example.org" not in cleaned
    assert ScientificTextCleaner.has_math(cleaned) is True
    assert ScientificTextCleaner.has_citation(cleaned) is True


def test_classical_profile_lowercases_and_strips_citations() -> None:
    cleaner = ScientificTextCleaner(profile=CleaningProfile.CLASSIFICATION_CLASSICAL)
    raw = "BERT [1, 2] uses $O(n^2)$ self-attention."
    cleaned = cleaner.clean(raw)

    assert cleaned == cleaned.lower()
    assert "[1, 2]" not in cleaned
    assert "[math]" in cleaned


def test_clean_qa_example_preserves_and_realigns_answer_offsets() -> None:
    # Leading whitespace + ligature 'ﬁ' (1 char -> 'fi' 2 chars) shifts answer_start!
    context = "  The ﬁnal model uses Vaswani attention for translation."
    answer = "Vaswani attention"
    orig_start = context.index(answer)

    cleaned = clean_qa_example(
        context=context,
        question="  Which   attention   mechanism is used? ",
        answer_text=answer,
        answer_start=orig_start,
    )

    assert cleaned["offset_valid"] is True
    start = cleaned["answer_start"]
    assert cleaned["context"][start : start + len(answer)] == answer
    assert cleaned["question"] == "Which attention mechanism is used?"


def test_clean_qa_example_handles_unanswerable() -> None:
    cleaned = clean_qa_example(
        context="Transformers rely on self-attention.",
        question="What is the batch size?",
        answer_text="",
        answer_start=-1,
    )
    assert cleaned["offset_valid"] is True
    assert cleaned["answer_text"] == ""
    assert cleaned["answer_start"] == -1
