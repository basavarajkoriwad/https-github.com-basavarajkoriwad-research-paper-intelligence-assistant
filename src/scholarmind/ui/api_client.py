"""Unified Streamlit-to-FastAPI client with automatic local in-process fallback."""

from __future__ import annotations

import os
import socket
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import requests

from scholarmind.api.dependencies import ServiceContainer
from scholarmind.api.routes.papers import ingest_uploaded_pdf
from scholarmind.api.routes.system import get_experiment_results, system_status
from scholarmind.core.config import load_config


class ScholarMindClient:
    """Connects the Streamlit dashboard to the FastAPI backend (or local engine fallback)."""

    def __init__(
        self,
        api_base_url: str | None = None,
        prefer_api: bool = True,
    ) -> None:
        raw_url = (api_base_url or os.getenv("SCHOLARMIND_API_URL", "http://127.0.0.1:8000")).strip()
        self.api_base_url = raw_url.rstrip("/")
        self.use_http_api = False
        self._container: ServiceContainer | None = None

        if prefer_api and self._is_api_reachable(self.api_base_url):
            self.use_http_api = True
        else:
            self._container = self._init_local_container()

    @staticmethod
    def _is_api_reachable(base_url: str) -> bool:
        try:
            parsed = urlparse(base_url)
            host = "127.0.0.1" if parsed.hostname in ("localhost", None) else parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            with socket.create_connection((host, port), timeout=0.35):
                pass
            resp = requests.get(f"{base_url}/health", timeout=1.5)
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _init_local_container() -> ServiceContainer:
        cfg = load_config()
        hf_cache = cfg.artifacts_dir / "hf_cache"
        if hf_cache.exists() and not os.getenv("HF_HOME"):
            os.environ["HF_HOME"] = str(hf_cache)
            os.environ.setdefault("HF_HUB_CACHE", str(hf_cache / "hub"))
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return ServiceContainer(config=cfg)

    @property
    def connection_label(self) -> str:
        if self.use_http_api:
            return f"FastAPI Connected ({self.api_base_url})"
        return "Local In-Process Engine (FastAPI offline fallback)"

    def _raise_for_api_error(self, resp: requests.Response) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"API Error ({resp.status_code}): {detail}")

    def get_system_status(self) -> dict[str, Any]:
        if self.use_http_api:
            resp = requests.get(f"{self.api_base_url}/system/status", timeout=15)
            self._raise_for_api_error(resp)
            return resp.json()
        assert self._container is not None
        return system_status(self._container)

    def get_experiment_results(self) -> dict[str, Any]:
        if self.use_http_api:
            resp = requests.get(f"{self.api_base_url}/experiments/results", timeout=15)
            self._raise_for_api_error(resp)
            return resp.json()
        assert self._container is not None
        return get_experiment_results(self._container)

    def list_documents(self) -> list[dict[str, Any]]:
        if self.use_http_api:
            resp = requests.get(f"{self.api_base_url}/documents", timeout=15)
            self._raise_for_api_error(resp)
            return resp.json().get("documents", [])
        assert self._container is not None
        raw = self._container.document_store.list_papers()
        return [
            {
                "document_id": p["paper_id"],
                "document_name": p["filename"],
                "title": p["title"],
                "authors": p["authors"],
                "page_count": p["page_count"],
                "word_count": p["word_count"],
                "chunk_count": p["chunk_count"],
                "ingested_at_utc": p["ingested_at_utc"],
            }
            for p in raw
        ]

    def get_paper_detail(self, document_id: str) -> dict[str, Any]:
        if self.use_http_api:
            resp = requests.get(f"{self.api_base_url}/api/v1/papers/{document_id}", timeout=15)
            self._raise_for_api_error(resp)
            return resp.json()
        assert self._container is not None
        paper = self._container.document_store.get_paper(document_id)
        if paper is None:
            raise ValueError(f"Document '{document_id}' not found.")
        return paper.model_dump()

    def upload_pdf(self, pdf_bytes: bytes, filename: str) -> dict[str, Any]:
        if self.use_http_api:
            resp = requests.post(
                f"{self.api_base_url}/upload",
                files={"file": (filename, pdf_bytes, "application/pdf")},
                timeout=120,
            )
            self._raise_for_api_error(resp)
            return resp.json()
        assert self._container is not None
        return ingest_uploaded_pdf(pdf_bytes=pdf_bytes, filename=filename, container=self._container).model_dump()

    def delete_document(self, document_id: str) -> dict[str, Any]:
        if self.use_http_api:
            resp = requests.delete(f"{self.api_base_url}/documents/{document_id}", timeout=30)
            self._raise_for_api_error(resp)
            return resp.json()
        assert self._container is not None
        deleted = self._container.document_store.delete_paper(document_id)
        removed_vecs = self._container.vector_store.delete_paper(document_id)
        return {"deleted": deleted, "document_id": document_id, "vectors_removed": removed_vecs}

    def summarize_paper(
        self,
        document_id: str,
        strategy: str = "auto",
        max_chunks: int = 4,
    ) -> dict[str, Any]:
        if self.use_http_api:
            resp = requests.post(
                f"{self.api_base_url}/api/v1/summarize/{document_id}",
                json={"strategy": strategy, "max_chunks": max_chunks},
                timeout=180,
            )
            self._raise_for_api_error(resp)
            return resp.json()
        assert self._container is not None
        paper = self._container.document_store.get_paper(document_id)
        if paper is None:
            raise ValueError(f"Document '{document_id}' not found.")
        chunks = self._container.document_store.get_chunks(document_id)
        return self._container.summarization_service.summarize_paper(
            paper=paper,
            chunks=chunks,
            strategy=strategy,
            max_chunks=max_chunks,
        ).model_dump()

    def ask_question(
        self,
        question: str,
        top_k: int = 4,
        mode: str = "hybrid",
        document_ids: list[str] | None = None,
        similarity_threshold: float = 0.34,
    ) -> dict[str, Any]:
        if self.use_http_api:
            resp = requests.post(
                f"{self.api_base_url}/ask",
                json={
                    "question": question,
                    "top_k": top_k,
                    "mode": mode,
                    "document_ids": document_ids,
                    "similarity_threshold": similarity_threshold,
                },
                timeout=120,
            )
            self._raise_for_api_error(resp)
            return resp.json()
        assert self._container is not None
        res = self._container.rag_service.answer(
            question=question,
            top_k=top_k,
            paper_ids=document_ids,
            mode=mode,
            similarity_threshold=similarity_threshold,
        )
        self._container.document_store.log_question(
            question=res.question,
            abstained=res.abstained,
            retrieval_status=res.retrieval_status,
            max_similarity=res.max_similarity_score,
            total_ms=res.latencies_ms.get("total_ms", 0.0),
            asked_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        return res.model_dump()

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        document_ids: list[str] | None = None,
        section_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if self.use_http_api:
            resp = requests.post(
                f"{self.api_base_url}/search",
                json={
                    "query": query,
                    "top_k": top_k,
                    "mode": mode,
                    "document_ids": document_ids,
                    "section_types": section_types,
                },
                timeout=60,
            )
            self._raise_for_api_error(resp)
            return resp.json()
        assert self._container is not None
        hits = self._container.search_engine.search(
            query=query,
            top_k=top_k,
            mode=mode,  # type: ignore[arg-type]
            paper_ids=document_ids,
            section_types=section_types,
        )
        return [h.model_dump() for h in hits]

    def analyze_paper(self, document_id: str, refresh: bool = False) -> dict[str, Any]:
        if self.use_http_api:
            resp = requests.get(
                f"{self.api_base_url}/api/v1/analysis/{document_id}",
                params={"refresh": str(refresh).lower()},
                timeout=60,
            )
            self._raise_for_api_error(resp)
            return resp.json()
        assert self._container is not None
        paper = self._container.document_store.get_paper(document_id)
        if paper is None:
            raise ValueError(f"Document '{document_id}' not found.")
        if not refresh:
            cached = self._container.document_store.get_analysis(document_id)
            if cached is not None:
                return cached.model_dump()
        report = self._container.analysis_service.analyze(paper)
        self._container.document_store.save_analysis(report)
        return report.model_dump()

    def compare_papers(self, document_ids: list[str]) -> dict[str, Any]:
        if self.use_http_api:
            resp = requests.post(
                f"{self.api_base_url}/api/v1/compare",
                json={"paper_ids": document_ids},
                timeout=90,
            )
            self._raise_for_api_error(resp)
            return resp.json()
        assert self._container is not None
        papers = []
        analyses = []
        for pid in document_ids:
            paper = self._container.document_store.get_paper(pid)
            if paper is None:
                raise ValueError(f"Document '{pid}' not found.")
            analysis = self._container.document_store.get_analysis(pid)
            if analysis is None:
                analysis = self._container.analysis_service.analyze(paper)
                self._container.document_store.save_analysis(analysis)
            papers.append(paper)
            analyses.append(analysis)
        return self._container.comparison_service.compare(papers, analyses).model_dump()
