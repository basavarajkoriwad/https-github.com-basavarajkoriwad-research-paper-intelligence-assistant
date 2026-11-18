"""End-to-end integration tests for FastAPI backend, real BGE-small FAISS RAG, T5 summarization, analysis & comparison."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
import pymupdf as fitz
import pytest

from scholarmind.api.dependencies import ServiceContainer
from scholarmind.api.main import create_app
from scholarmind.core.config import load_config


def _build_pdf(pages: list[str], title: str, author: str) -> bytes:
    doc = fitz.open()
    doc.set_metadata({"title": title, "author": author})
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 780), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


PAPER_A_PAGES = [
    (
        "Abstract\n"
        "We propose SparseSciFormer, a hybrid attention Transformer architecture for scientific paper summarization. "
        "We evaluate SparseSciFormer on ArXiv and PubMed benchmarks using ROUGE-1 and ROUGE-L metrics. "
        "Our model achieves 44.2% ROUGE-1 and outperforms standard BERT and T5 baselines by 3.8%.\n\n"
        "1. Introduction\n"
        "Scientific papers routinely exceed 6,000 tokens, causing severe truncation in standard Transformer encoders. "
        "Our objective is to process full research documents efficiently without losing methodological details."
    ),
    (
        "2. Methodology\n"
        "SparseSciFormer combines local sliding-window self-attention with global section anchor tokens. "
        "We fine-tune the T5 encoder-decoder using LoRA adapters, AdamW optimizer, and cross-entropy loss across 8 epochs.\n\n"
        "3. Results\n"
        "On the ArXiv dataset, SparseSciFormer achieves 44.2% ROUGE-1 and 39.1% ROUGE-L while reducing GPU memory by 42%."
    ),
    (
        "4. Limitations\n"
        "However, SparseSciFormer has not been evaluated on non-English corpora and still struggles with dense multi-page equations. "
        "In addition, inference latency remains high on CPU-only laptops.\n\n"
        "5. Conclusion and Future Work\n"
        "In future work, we plan to extend our framework to multimodal figure-table understanding and quantized ONNX deployment."
    ),
]

PAPER_B_PAGES = [
    (
        "Abstract\n"
        "We introduce HybridSciRAG, a retrieval-augmented generation framework for grounded scientific question answering. "
        "We benchmark HybridSciRAG on ArXiv, SciQ, and SQuAD 2.0 datasets using Recall@10 and F1 scores. "
        "Results show that fusing BGE dense embeddings with BM25 lexical scores improves Recall@10 by 11.4%.\n\n"
        "1. Methodology\n"
        "Our pipeline indexes section-aware chunks into FAISS using BGE embeddings and ranks passages via Reciprocal Rank Fusion. "
        "A SciBERT extractive verifier checks answerability before invoking a Flan-T5 generator."
    ),
    (
        "2. Results\n"
        "HybridSciRAG achieves 96.5% Recall@1 on SciQ and 84.7% F1 on SQuAD 2.0 unanswerable detection.\n\n"
        "3. Limitations\n"
        "HybridSciRAG fails to parse severely corrupted scanned PDFs without OCR and exhibits limited cross-table numerical reasoning. "
        "We leave multimodal table reasoning for future work."
    ),
]


@pytest.mark.slow_model
def test_fastapi_backend_end_to_end_with_real_models(tmp_path: Path) -> None:
    cfg = load_config()
    hf_cache = cfg.artifacts_dir / "hf_cache"
    if hf_cache.exists():
        os.environ["HF_HOME"] = str(hf_cache)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    container = ServiceContainer(config=cfg, store_dir=tmp_path / "store")
    app = create_app(container_override=container)

    async def _run_all_endpoint_checks() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1. Health check
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"
            assert health.json()["indexed_papers"] == 0

            # 2. Error handling: non-PDF and corrupted PDF
            bad_ext = await client.post(
                "/api/v1/papers/upload",
                files={"file": ("notes.txt", b"hello world", "text/plain")},
            )
            assert bad_ext.status_code == 400

            corrupt_pdf = await client.post(
                "/api/v1/papers/upload",
                files={"file": ("broken.pdf", b"%PDF-1.7\ncorrupted", "application/pdf")},
            )
            assert corrupt_pdf.status_code == 400
            assert corrupt_pdf.json()["error"] == "pdf_corrupted"

            # 3. Upload Paper A & Paper B
            pdf_a = _build_pdf(
                PAPER_A_PAGES,
                title="SparseSciFormer: Efficient Scientific Summarization",
                author="A. Author",
            )
            pdf_b = _build_pdf(
                PAPER_B_PAGES,
                title="HybridSciRAG: Grounded Scientific QA",
                author="B. Author",
            )

            up_a = await client.post(
                "/api/v1/papers/upload",
                files={"file": ("sparse_sciformer.pdf", pdf_a, "application/pdf")},
            )
            assert up_a.status_code == 201
            paper_a_id = up_a.json()["paper_id"]
            assert up_a.json()["page_count"] == 3
            assert up_a.json()["chunks_indexed"] >= 3

            up_b = await client.post(
                "/api/v1/papers/upload",
                files={"file": ("hybrid_scirag.pdf", pdf_b, "application/pdf")},
            )
            assert up_b.status_code == 201
            paper_b_id = up_b.json()["paper_id"]
            assert up_b.json()["page_count"] == 2

            # 4. List & Get Paper
            listing = await client.get("/api/v1/papers")
            assert listing.status_code == 200
            assert listing.json()["count"] == 2

            detail_a = await client.get(f"/api/v1/papers/{paper_a_id}")
            assert detail_a.status_code == 200
            assert len(detail_a.json()["pages"]) == 3

            assert (await client.get("/api/v1/papers/nonexistent_id")).status_code == 404

            # 5. Semantic Search (Hybrid FAISS + BM25)
            search_resp = await client.post(
                "/api/v1/search",
                json={
                    "query": "What ROUGE-1 score did SparseSciFormer achieve on ArXiv?",
                    "top_k": 3,
                    "mode": "hybrid",
                },
            )
            assert search_resp.status_code == 200
            hits = search_resp.json()
            assert len(hits) >= 1
            assert hits[0]["paper_id"] == paper_a_id
            assert hits[0]["page_number"] in (1, 2)

            # 6. Grounded RAG Q&A — Supported Question
            qa_supported = await client.post(
                "/api/v1/qa",
                json={
                    "question": "How much did SparseSciFormer reduce GPU memory on ArXiv?",
                    "top_k": 3,
                },
            )
            assert qa_supported.status_code == 200
            qa_body = qa_supported.json()
            assert qa_body["grounded"] is True
            assert qa_body["abstained"] is False
            assert len(qa_body["citations"]) >= 1
            assert qa_body["citations"][0]["paper_id"] == paper_a_id
            assert qa_body["citations"][0]["page_number"] >= 1
            assert "Sources:" in qa_body["answer"]

            # 7. Grounded RAG Q&A — Out-of-Scope Question (Anti-Hallucination Abstention!)
            qa_oob = await client.post(
                "/api/v1/qa",
                json={
                    "question": "What is the authentic Italian recipe for tiramisu dessert?",
                    "top_k": 3,
                },
            )
            assert qa_oob.status_code == 200
            oob_body = qa_oob.json()
            assert oob_body["abstained"] is True
            assert oob_body["citations"] == []
            assert "sufficient information" in oob_body["answer"].lower()

            # 8. T5 Summarization (reusing Experiment D fine-tuned checkpoint + hierarchical chunking!)
            summ_resp = await client.post(
                f"/api/v1/summarize/{paper_a_id}",
                json={"strategy": "hierarchical", "max_chunks": 3},
            )
            assert summ_resp.status_code == 200
            summ_body = summ_resp.json()
            assert summ_body["strategy"] == "hierarchical"
            assert summ_body["chunks_processed"] >= 2
            assert len(summ_body["summary"]) > 20

            # 9. Grounded Research Analysis
            ana_a = await client.get(f"/api/v1/analysis/{paper_a_id}")
            assert ana_a.status_code == 200
            ana_body = ana_a.json()
            assert "ArXiv" in ana_body["datasets"]
            assert "PubMed" in ana_body["datasets"]
            assert "T5" in ana_body["model_architectures"]
            assert "LoRA" in ana_body["model_architectures"]
            assert len(ana_body["objectives"]) >= 1
            assert len(ana_body["methodologies"]) >= 1
            assert len(ana_body["key_findings"]) >= 1
            assert len(ana_body["explicit_limitations"]) >= 1
            assert len(ana_body["future_directions"]) >= 1

            # 10. Multi-Paper Comparison
            cmp_resp = await client.post(
                "/api/v1/compare",
                json={"paper_ids": [paper_a_id, paper_b_id]},
            )
            assert cmp_resp.status_code == 200
            cmp_body = cmp_resp.json()
            assert len(cmp_body["comparison_matrix"]) == 2
            assert "ArXiv" in cmp_body["shared_datasets"]
            assert len(cmp_body["citations"]) >= 2
            assert "SparseSciFormer" in cmp_body["comparative_summary"]

            # Validation check: < 2 papers rejected cleanly
            assert (await client.post("/api/v1/compare", json={"paper_ids": [paper_a_id]})).status_code == 422

            # 11. Delete Paper & Verify Cleanup
            del_resp = await client.delete(f"/api/v1/papers/{paper_b_id}")
            assert del_resp.status_code == 200
            assert del_resp.json()["deleted"] is True
            assert del_resp.json()["vectors_removed"] >= 2
            assert (await client.get("/api/v1/papers")).json()["count"] == 1

    asyncio.run(_run_all_endpoint_checks())
