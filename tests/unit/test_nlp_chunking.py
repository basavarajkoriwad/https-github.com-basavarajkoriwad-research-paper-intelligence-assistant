"""Unit tests for scientific section detection and hierarchical document chunking."""

from __future__ import annotations

from scholarmind.nlp.chunking import (
    FixedWindowChunker,
    HierarchicalSectionChunker,
    ScientificSectionDetector,
)

SAMPLE_PAPER = """
Abstract
We propose ScholarNet, a sparse attention architecture for scientific literature understanding.

1. Introduction
Scientific papers are long and dense. Prior models struggle with 512-token limits [1].
Vaswani et al. introduced self-attention, which scales quadratically as $O(n^2)$.

2. Methodology
Our model partitions documents into hierarchical sections.
Each token attends locally within its paragraph and globally to section anchors.
We optimize cross-entropy loss with AdamW and cosine warmup across 8 epochs.
To ensure scalability, we apply gradient checkpointing and mixed precision training.

3. Limitations
Our method has not been evaluated on non-English biomedical corpora.
Furthermore, extreme mathematical derivations with complex matrices remain challenging.

4. Conclusion
ScholarNet improves scientific retrieval and summarization accuracy while lowering compute cost.
""".strip()


def test_section_detector_identifies_canonical_sections() -> None:
    detector = ScientificSectionDetector()
    sections = detector.split_into_sections(SAMPLE_PAPER, paper_id="paper_001")

    types = [s.section_type for s in sections]
    assert "abstract" in types
    assert "introduction" in types
    assert "methodology" in types
    assert "limitations" in types
    assert "conclusion" in types


def test_hierarchical_chunker_creates_linked_parent_and_child_chunks() -> None:
    chunker = HierarchicalSectionChunker(
        max_tokens=45,
        overlap_sentences=1,
        parent_max_tokens=300,
        include_breadcrumb=True,
    )
    chunks = chunker.chunk_document(
        text=SAMPLE_PAPER,
        paper_id="paper_001",
        paper_title="ScholarNet: Sparse Attention for Scientific Papers",
        include_parents=True,
    )

    parents = [c for c in chunks if c.chunk_level == "parent"]
    children = [c for c in chunks if c.chunk_level == "child"]

    assert len(parents) >= 4
    assert len(children) >= len(parents)

    parent_ids = {p.chunk_id for p in parents}
    for child in children:
        assert child.parent_id in parent_ids
        assert child.contextualized_text.startswith("[Paper: ScholarNet")
        assert child.estimated_tokens > 0

    # Verify section metadata & math/citation detection preserved
    intro_children = [c for c in children if c.section_type == "introduction"]
    assert any(c.has_math for c in intro_children)
    assert any(c.has_citation for c in intro_children)

    lim_children = [c for c in children if c.section_type == "limitations"]
    assert len(lim_children) >= 1
    assert any("not been evaluated" in c.text for c in lim_children)


def test_fixed_window_chunker_baseline() -> None:
    chunker = FixedWindowChunker(window_words=20, overlap_words=5)
    chunks = chunker.chunk_document(SAMPLE_PAPER, paper_id="paper_001")
    assert len(chunks) >= 3
    assert all(c.word_count <= 20 for c in chunks)
