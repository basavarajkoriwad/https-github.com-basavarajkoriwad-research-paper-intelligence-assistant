"""FastAPI application factory with structured logging, CORS, and error handling."""

from __future__ import annotations

import os
import time

from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from scholarmind import __version__
from scholarmind.api.dependencies import ServiceContainer, get_container
from scholarmind.api.routes import analysis, module5, papers, qa, search, summarize, system
from scholarmind.api.schemas import HealthResponse
from scholarmind.core.logging import get_logger
from scholarmind.ingestion.pdf_parser import (
    PDFCorruptedError,
    PDFTooLargeError,
    PDFValidationError,
)

logger = get_logger(__name__)


def create_app(container_override: ServiceContainer | None = None) -> FastAPI:
    """Create and configure the ScholarMind AI FastAPI application."""
    application = FastAPI(
        title="ScholarMind AI — Research Paper Intelligence API",
        description=(
            "Modular scientific PDF intelligence, FAISS + BM25 hybrid retrieval, "
            "grounded RAG Q&A, T5 summarization, research analysis, and multi-paper comparison."
        ),
        version=__version__,
    )

    raw_origins = os.getenv("SCHOLARMIND_CORS_ORIGINS", "*").strip()
    allow_origins = [o.strip() for o in raw_origins.split(",") if o.strip()] or ["*"]

    application.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if container_override is not None:
        application.dependency_overrides[get_container] = lambda: container_override

    @application.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "%s %s -> %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @application.exception_handler(PDFTooLargeError)
    async def handle_pdf_too_large(_request: Request, exc: PDFTooLargeError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={"error": "pdf_too_large", "detail": str(exc)},
        )

    @application.exception_handler(PDFCorruptedError)
    async def handle_pdf_corrupted(_request: Request, exc: PDFCorruptedError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "pdf_corrupted", "detail": str(exc)},
        )

    @application.exception_handler(PDFValidationError)
    async def handle_pdf_validation(_request: Request, exc: PDFValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "pdf_invalid", "detail": str(exc)},
        )

    @application.exception_handler(ValueError)
    async def handle_value_error(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "bad_request", "detail": str(exc)},
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server exception: %s", exc)
        is_prod = os.getenv("SCHOLARMIND_ENV", "development").lower() == "production"
        detail = "An internal server error occurred." if is_prod else str(exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "internal_server_error", "detail": detail},
        )

    @application.get("/", tags=["System"])
    def root() -> dict[str, str]:
        return {
            "service": "ScholarMind AI — Research Paper Intelligence API",
            "version": __version__,
            "status": "running",
            "api_docs": "/docs",
            "health": "/health",
            "system_status": "/system/status",
            "streamlit_ui_hint": "Run `.venv\\Scripts\\python.exe scripts\\run_ui.py` and open http://localhost:8501",
        }

    @application.get("/health", response_model=HealthResponse, tags=["System"])
    def health_check(container: ServiceContainer = Depends(get_container)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            service="ScholarMind AI",
            version=__version__,
            indexed_papers=len(container.document_store.list_papers()),
            indexed_vectors=container.vector_store.size,
            llm_backend=container.llm_client.backend_name,
        )

    application.include_router(module5.router)
    application.include_router(system.router)
    application.include_router(papers.router)
    application.include_router(search.router)
    application.include_router(qa.router)
    application.include_router(summarize.router)
    application.include_router(analysis.router)

    return application


app = create_app()
