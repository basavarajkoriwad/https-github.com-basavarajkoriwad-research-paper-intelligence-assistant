"""Unit tests for PDF parsing, SQLite persistence, FAISS persistence, and Hybrid Search."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf as fitz
import pytest

from scholarmind.ingestion.document_store import SQLiteDocumentStore
from scholarmind.ingestion.pdf_parser import (
    PDFCorruptedError,
    PDFTooLargeError,
    PDFValidationError,
    PyMuPDFParser,
)
from scholarmind.models.embeddings.retrieval_benchmark import BaseTextEmbedder
from scholarmind.retrieval.hybrid_search import HybridSearchEngine
from scholarmind.retrieval.vector_store import FAISSVectorStore


def make_synthetic_pdf(pages: list[str], title: str = "", author: str = "") -> bytes:
    """Create a valid multi-page PDF byte stream using PyMuPDF."""
    doc = fitz.open()
    if title or author:
        doc.set_metadata({"title": title, "author": author})
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 780), text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class _HashEmbedder(BaseTextEmbedder):
    """Fast deterministic bag-of-words projection embedder for unit tests."""

    name = "hash-unit-embedder"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                clean = "".join(ch for ch in tok if ch.isalnum())
                if clean:
                    vecs[i, hash(clean) % self.dim] += 1.0
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm
        return vecs


def test_pdf_parser_preserves_pages_sections_and_handles_truncation() -> None:
    pdf_bytes = make_synthetic_pdf(
        pages=[
            "Abstract\nWe propose SciGraphFormer to improve citation retrieval on ArXiv and SciQ.\n\n"
            "1. Introduction\nResearch papers span thousands of tokens and challenge standard BERT encoders.",
            "2. Methodology\nOur framework combines SciBERT embeddings with FAISS dense retrieval and BM25 fusion.",
            "3. Limitations\nOur method has not been evaluated on biomedical clinical trials and exhibits high memory usage.",
        ],
        title="SciGraphFormer: Hybrid Scientific Retrieval",
        author="A. Researcher, B. Scientist",
    )

    parser = PyMuPDFParser(max_pages=10, max_tokens_per_chunk=120)
    parsed, chunks = parser.parse_bytes(pdf_bytes, filename="scigraphformer.pdf")

    assert parsed.title == "SciGraphFormer: Hybrid Scientific Retrieval"
    assert parsed.authors == "A. Researcher, B. Scientist"
    assert parsed.page_count == 3
    assert parsed.pages_processed == 3
    assert parsed.truncated is False
    assert {c.page_number for c in chunks} == {1, 2, 3}
    assert any(c.section_type == "limitations" and c.page_number == 3 for c in chunks)

    # Large PDF guard: cap to 2 pages -> truncated=True
    capped_parser = PyMuPDFParser(max_pages=2)
    parsed_capped, capped_chunks = capped_parser.parse_bytes(pdf_bytes, filename="large.pdf")
    assert parsed_capped.page_count == 3
    assert parsed_capped.pages_processed == 2
    assert parsed_capped.truncated is True
    assert max(c.page_number for c in capped_chunks) == 2


def test_pdf_parser_rejects_corrupted_empty_and_oversized_files() -> None:
    parser = PyMuPDFParser(max_file_size_mb=0.001)  # ~1 KB limit

    with pytest.raises(PDFValidationError):
        parser.parse_bytes(b"", filename="empty.pdf")

    with pytest.raises(PDFTooLargeError):
        parser.parse_bytes(b"%PDF-1.7\n" + b"x" * 3000, filename="huge.pdf")

    normal_parser = PyMuPDFParser(max_file_size_mb=5.0)
    with pytest.raises(PDFCorruptedError):
        normal_parser.parse_bytes(b"NOT A PDF AT ALL", filename="bad.pdf")

    with pytest.raises(PDFCorruptedError):
        normal_parser.parse_bytes(b"%PDF-1.7\ncorrupted garbage bytes", filename="corrupt.pdf")


def test_sqlite_faiss_persistence_and_hybrid_search(tmp_path: Path) -> None:
    pdf1 = make_synthetic_pdf(
        pages=[
            "Abstract\nWe propose LoRA-SciT5 for parameter-efficient ArXiv summarization using T5 and ROUGE.",
            "2. Limitations\nTraining requires GPU memory and struggles on mathematical proofs.",
        ],
        title="LoRA-SciT5 Paper",
    )
    pdf2 = make_synthetic_pdf(
        pages=[
            "Abstract\nWe study protein folding thermodynamics using molecular dynamics simulations.",
        ],
        title="Protein Dynamics Paper",
    )

    parser = PyMuPDFParser()
    p1, c1 = parser.parse_bytes(pdf1, filename="p1.pdf")
    p2, c2 = parser.parse_bytes(pdf2, filename="p2.pdf")

    # 1. SQLite persistence
    store = SQLiteDocumentStore(db_path=tmp_path / "docs.db")
    store.save_paper(p1, c1)
    store.save_paper(p2, c2)
    assert len(store.list_papers()) == 2
    assert store.get_paper(p1.paper_id) is not None
    assert len(store.get_chunks(p1.paper_id)) == len(c1)

    # 2. FAISS persistence across reload
    embedder = _HashEmbedder(dim=64)
    faiss_dir = tmp_path / "faiss"
    vstore = FAISSVectorStore(index_dir=faiss_dir, embedder=embedder)
    vstore.add_chunks(c1)
    vstore.add_chunks(c2)
    assert vstore.size == len(c1) + len(c2)

    reloaded = FAISSVectorStore(index_dir=faiss_dir, embedder=embedder)
    assert reloaded.size == vstore.size

    # 3. Hybrid Search (dense, lexical, hybrid, metadata filtering)
    engine = HybridSearchEngine(vector_store=reloaded)
    hits_hybrid = engine.search("ArXiv summarization T5 ROUGE", top_k=3, mode="hybrid")
    assert len(hits_hybrid) >= 1
    assert hits_hybrid[0].paper_id == p1.paper_id
    assert hits_hybrid[0].page_number == 1

    hits_lim = engine.search("mathematical proofs GPU memory", top_k=3, section_types=["limitations"])
    assert len(hits_lim) == 1
    assert hits_lim[0].page_number == 2

    # 4. Deletion consistency
    removed = reloaded.delete_paper(p2.paper_id)
    assert removed == len(c2)
    assert reloaded.size == len(c1)
    assert store.delete_paper(p2.paper_id) is True
    assert len(store.list_papers()) == 1
