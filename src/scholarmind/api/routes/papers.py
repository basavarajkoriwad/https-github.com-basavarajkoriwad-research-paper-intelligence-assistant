"""Paper upload, listing, inspection, and deletion routes (with unchanged-document dedup)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from scholarmind.api.dependencies import ServiceContainer, get_container
from scholarmind.api.schemas import PaperUploadResponse
from scholarmind.core.schemas import ParsedPaper

router = APIRouter(prefix="/api/v1/papers", tags=["Papers"])


def _sanitize_filename(raw_name: str) -> str:
    """Strip any directory components or unsafe characters from uploaded filenames."""
    base = Path(raw_name or "uploaded.pdf").name
    safe = re.sub(r"[^A-Za-z0-9._\-\s]", "_", base).strip()
    return safe or "uploaded.pdf"


def _uploads_dir(container: ServiceContainer) -> Path:
    up_dir = container.document_store.db_path.parent / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    return up_dir


def delete_archived_pdf(paper_id: str, container: ServiceContainer) -> None:
    """Remove safely archived raw PDF if present."""
    pdf_path = _uploads_dir(container) / f"{paper_id}.pdf"
    if pdf_path.exists():
        pdf_path.unlink()


def ingest_uploaded_pdf(
    pdf_bytes: bytes,
    filename: str,
    container: ServiceContainer,
) -> PaperUploadResponse:
    """Shared PDF ingestion routine avoiding reprocessing of unchanged already-indexed PDFs."""
    safe_filename = _sanitize_filename(filename)
    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files (.pdf) are supported.",
        )

    # Content-based deterministic ID avoids recomputing embeddings for unchanged documents
    if pdf_bytes:
        candidate_id = f"paper_{hashlib.sha256(pdf_bytes).hexdigest()[:12]}"
        if container.document_store.paper_exists(candidate_id) and container.vector_store.has_paper(candidate_id):
            existing = container.document_store.get_paper(candidate_id)
            if existing is not None:
                existing_chunks = container.document_store.get_chunks(candidate_id)
                unique_sections = list(dict.fromkeys(sec.section_title for sec in existing.sections))
                return PaperUploadResponse(
                    document_id=existing.paper_id,
                    document_name=existing.filename,
                    paper_id=existing.paper_id,
                    filename=existing.filename,
                    title=existing.title,
                    authors=existing.authors,
                    page_count=existing.page_count,
                    pages_processed=existing.pages_processed,
                    truncated=existing.truncated,
                    word_count=existing.word_count,
                    chunks_indexed=len(existing_chunks),
                    already_indexed=True,
                    sections_detected=unique_sections[:20],
                )

    parsed, chunks = container.pdf_parser.parse_bytes(pdf_bytes=pdf_bytes, filename=safe_filename)
    container.document_store.save_paper(parsed, chunks)
    indexed_count = container.vector_store.add_chunks(chunks)

    # Safely archive validated PDF bytes under isolated uploads/<paper_id>.pdf
    archive_path = _uploads_dir(container) / f"{parsed.paper_id}.pdf"
    archive_path.write_bytes(pdf_bytes)

    unique_sections = list(dict.fromkeys(sec.section_title for sec in parsed.sections))
    return PaperUploadResponse(
        document_id=parsed.paper_id,
        document_name=parsed.filename,
        paper_id=parsed.paper_id,
        filename=parsed.filename,
        title=parsed.title,
        authors=parsed.authors,
        page_count=parsed.page_count,
        pages_processed=parsed.pages_processed,
        truncated=parsed.truncated,
        word_count=parsed.word_count,
        chunks_indexed=indexed_count,
        already_indexed=False,
        sections_detected=unique_sections[:20],
    )


@router.post("/upload", response_model=PaperUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_paper(
    file: UploadFile = File(...),
    container: ServiceContainer = Depends(get_container),
) -> PaperUploadResponse:
    """Upload a scientific PDF, extract page-preserving text/chunks, and persist to SQLite + FAISS."""
    filename = file.filename or "uploaded.pdf"
    pdf_bytes = await file.read()
    return ingest_uploaded_pdf(pdf_bytes=pdf_bytes, filename=filename, container=container)


@router.get("")
def list_papers(container: ServiceContainer = Depends(get_container)) -> dict:
    """List all uploaded and persisted research papers."""
    papers = container.document_store.list_papers()
    return {"count": len(papers), "papers": papers}


@router.get("/{paper_id}", response_model=ParsedPaper)
def get_paper(
    paper_id: str,
    container: ServiceContainer = Depends(get_container),
) -> ParsedPaper:
    """Retrieve parsed metadata, pages, and sections for a specific paper."""
    paper = container.document_store.get_paper(paper_id)
    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper '{paper_id}' not found.",
        )
    return paper


@router.delete("/{paper_id}")
def delete_paper(
    paper_id: str,
    container: ServiceContainer = Depends(get_container),
) -> dict:
    """Delete a paper from SQLite and remove its vectors from the persisted FAISS index."""
    deleted = container.document_store.delete_paper(paper_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper '{paper_id}' not found.",
        )
    removed_vectors = container.vector_store.delete_paper(paper_id)
    delete_archived_pdf(paper_id, container)
    return {
        "deleted": True,
        "document_id": paper_id,
        "paper_id": paper_id,
        "vectors_removed": removed_vectors,
    }
