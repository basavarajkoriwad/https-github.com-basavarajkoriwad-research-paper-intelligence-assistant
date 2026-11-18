"""Fast, self-contained CI integration tests for FastAPI backend, FAISS, SQLite, and RAG (no large HF downloads)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import numpy as np
import pymupdf as fitz

from scholarmind.api.dependencies import ServiceContainer
from scholarmind.api.main import create_app
from scholarmind.models.embeddings.retrieval_benchmark import BaseTextEmbedder
from scholarmind.models.llm.open_llm import BaseLLMClient


class _DeterministicCIEmbedder(BaseTextEmbedder):
    """Lightweight deterministic BoW embedder so CI validates FAISS math without downloading GBs."""

    name = "ci-deterministic-embedder"

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                clean = "".join(ch for ch in tok if ch.isalnum())
                if len(clean) >= 2:
                    vecs[i, sum(ord(c) * (idx + 1) for idx, c in enumerate(clean)) % self.dim] += 1.0
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm
        return vecs


class _DeterministicCILLM(BaseLLMClient):
    """Lightweight deterministic grounded responder for CI workflow validation."""

    backend_name = "ci-deterministic-llm"

    def generate(self, prompt: str, max_new_tokens: int = 160) -> str:
        if "ArXiv" in prompt and "SciQ" in prompt:
            return "The paper evaluates its architecture on the ArXiv and SciQ datasets using FAISS."
        return "The uploaded documents do not provide sufficient information."


def _build_pdf(pages: list[str], title: str) -> bytes:
    doc = fitz.open()
    doc.set_metadata({"title": title, "author": "CI Tester"})
    for text in pages:
        p = doc.new_page()
        p.insert_textbox(fitz.Rect(50, 50, 545, 780), text, fontsize=11)
    raw = doc.tobytes()
    doc.close()
    return raw


def test_ci_full_backend_api_and_security_guards(tmp_path: Path) -> None:
    """Validates every core API route, filename sanitization, persistence, RAG abstention & cleanup."""
    container = ServiceContainer(
        store_dir=tmp_path / "ci_store",
        embedder=_DeterministicCIEmbedder(),
        llm_client=_DeterministicCILLM(),
    )
    app = create_app(container_override=container)

    pdf_a = _build_pdf(
        [
            "Abstract\nWe propose CISparseNet evaluated on ArXiv and SciQ datasets using FAISS and Transformer layers.",
            "2. Limitations\nCISparseNet cannot process scanned image PDFs without OCR and has high memory overhead.",
        ],
        title="CISparseNet Paper",
    )
    pdf_b = _build_pdf(
        [
            "Abstract\nWe introduce CIBaselineRAG evaluated on ArXiv and PubMed using BM25 and T5.",
        ],
        title="CIBaselineRAG Paper",
    )

    async def _run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            root = await client.get("/")
            assert root.status_code == 200
            assert root.json()["status"] == "running"

            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"

            # Path traversal filename must be safely sanitized
            up_a = await client.post(
                "/upload",
                files={"file": ("../../evil/CISparseNet.pdf", pdf_a, "application/pdf")},
            )
            assert up_a.status_code == 201
            doc_a = up_a.json()["document_id"]
            assert up_a.json()["document_name"] == "CISparseNet.pdf"
            assert (tmp_path / "ci_store" / "uploads" / f"{doc_a}.pdf").exists()

            # Duplicate upload must skip recomputation
            up_a_dup = await client.post(
                "/upload",
                files={"file": ("CISparseNet.pdf", pdf_a, "application/pdf")},
            )
            assert up_a_dup.status_code == 201
            assert up_a_dup.json()["already_indexed"] is True

            up_b = await client.post(
                "/upload",
                files={"file": ("CIBaselineRAG.pdf", pdf_b, "application/pdf")},
            )
            assert up_b.status_code == 201
            doc_b = up_b.json()["document_id"]

            # Search
            sr = await client.post("/search", json={"query": "ArXiv SciQ FAISS", "top_k": 3})
            assert sr.status_code == 200
            assert len(sr.json()) >= 1

            # Grounded Q&A (supported)
            ask_ok = await client.post(
                "/ask",
                json={
                    "question": "Which datasets ArXiv SciQ FAISS are used?",
                    "top_k": 3,
                    "similarity_threshold": 0.15,
                },
            )
            assert ask_ok.status_code == 200
            assert ask_ok.json()["abstained"] is False
            assert ask_ok.json()["sources"][0]["document_name"] == "CISparseNet.pdf"

            # Grounded Q&A (out-of-scope abstention)
            ask_oob = await client.post(
                "/ask",
                json={"question": "What is the recipe for banana pancakes?", "top_k": 3},
            )
            assert ask_oob.status_code == 200
            assert ask_oob.json()["abstained"] is True

            # Analysis & Multi-paper Comparison
            ana = await client.get(f"/api/v1/analysis/{doc_a}")
            assert ana.status_code == 200
            assert "ArXiv" in ana.json()["datasets"]

            cmp_r = await client.post("/api/v1/compare", json={"paper_ids": [doc_a, doc_b]})
            assert cmp_r.status_code == 200
            assert "ArXiv" in cmp_r.json()["shared_datasets"]

            # System telemetry check
            st = await client.get("/system/status")
            assert st.status_code == 200
            assert st.json()["indexed_papers"] == 2
            assert st.json()["qa_stats"]["questions_asked"] == 2

            # Synchronized deletion (SQLite + FAISS + archived PDF)
            dl = await client.delete(f"/documents/{doc_a}")
            assert dl.status_code == 200
            assert not (tmp_path / "ci_store" / "uploads" / f"{doc_a}.pdf").exists()

    asyncio.run(_run())
