"""Unit tests for scientific sentence segmentation, tokenization, vocab, and normalization."""

from __future__ import annotations

from pathlib import Path

from scholarmind.nlp.normalization import (
    ScientificNormalizer,
    should_apply_classical_normalization,
)
from scholarmind.nlp.segmentation import ScientificSentenceSegmenter
from scholarmind.nlp.tokenization import (
    ScientificTokenizer,
    Vocabulary,
    estimate_subword_tokens,
)


def test_sentence_segmenter_protects_scientific_abbreviations() -> None:
    segmenter = ScientificSentenceSegmenter(backend="auto")
    text = (
        "Vaswani et al. introduced the Transformer in Sec. 3.2 (see Fig. 1 and Eq. 4). "
        "It achieved 28.4 BLEU, e.g. outperforming RNNs vs. prior baselines. "
        "However, quadratic complexity remains a limitation!"
    )
    sents = segmenter.segment(text)

    assert len(sents) == 3
    assert "Vaswani et al." in sents[0]
    assert "Sec. 3.2" in sents[0]
    assert "28.4 BLEU" in sents[1]
    assert sents[2].startswith("However")


def test_scientific_tokenizer_preserves_hyphenated_terms_and_placeholders() -> None:
    tok = ScientificTokenizer(backend="scientific_regex", lowercase=False, strip_punctuation=True)
    tokens = tok.tokenize("DeBERTa-v3 uses self-attention with [MATH] and F1-score 92.4.")

    assert "DeBERTa-v3" in tokens
    assert "self-attention" in tokens
    assert "[MATH]" in tokens
    assert "F1-score" in tokens
    assert "92.4" in tokens
    assert "." not in tokens


def test_vocabulary_fit_encode_decode_roundtrip(tmp_path: Path) -> None:
    corpus = [
        ["transformer", "attention", "model"],
        ["transformer", "layer", "rareword"],
        ["attention", "model"],
    ]
    vocab = Vocabulary(min_freq=2, max_size=100).fit(corpus)

    assert "transformer" in vocab.token_to_id
    assert "attention" in vocab.token_to_id
    assert "rareword" not in vocab.token_to_id  # frequency 1 filtered out

    encoded = vocab.encode(["transformer", "rareword", "model"], add_bos=True, add_eos=True)
    assert encoded[0] == vocab.token_to_id[Vocabulary.BOS_TOKEN]
    assert encoded[2] == vocab.unk_id
    assert vocab.decode(encoded, skip_special=True) == ["transformer", "model"]

    vocab_path = tmp_path / "vocab.json"
    vocab.save(vocab_path)
    loaded = Vocabulary.load(vocab_path)
    assert loaded.token_to_id == vocab.token_to_id


def test_normalizer_preserves_negations_and_policy_guardrails() -> None:
    normalizer = ScientificNormalizer(backend="auto", preserve_negations=True)
    tokens = normalizer.normalize_text(
        "The baseline models did not generalize without pretraining, however results improved."
    )

    assert "not" in tokens
    assert "without" in tokens
    assert "however" in tokens
    assert "the" not in tokens

    # Policy check: classical normalization is ONLY for lexical/classical pipelines
    assert should_apply_classical_normalization("bm25") is True
    assert should_apply_classical_normalization("tfidf") is True
    assert should_apply_classical_normalization("bert_qa") is False
    assert should_apply_classical_normalization("t5_summarization") is False
    assert should_apply_classical_normalization("rag_llm") is False


def test_estimate_subword_tokens() -> None:
    assert estimate_subword_tokens("") == 0
    assert estimate_subword_tokens("neural network") >= 2
