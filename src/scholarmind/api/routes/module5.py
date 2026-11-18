"""Module 5 canonical root RAG API routes (/upload, /ask, /search, /documents)."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from scholarmind.api.dependencies import ServiceContainer, get_container
from scholarmind.api.routes.papers import delete_archived_pdf, ingest_uploaded_pdf
from scholarmind.api.schemas import PaperUploadResponse, QARequest, SearchRequest
from scholarmind.core.schemas import RAGAnswerResponse, RetrievedPassage

router = APIRouter(tags=["Module 5 RAG API"])


@router.post("/upload", response_model=PaperUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    container: ServiceContainer = Depends(get_container),
) -> PaperUploadResponse:
    """Upload a scientific research PDF, chunk with page metadata, and index in FAISS."""
    filename = file.filename or "uploaded.pdf"
    pdf_bytes = await file.read()
    return ingest_uploaded_pdf(pdf_bytes=pdf_bytes, filename=filename, container=container)


@router.post("/ask", response_model=RAGAnswerResponse)
def ask_rag(
    request: QARequest,
    container: ServiceContainer = Depends(get_container),
) -> RAGAnswerResponse:
    """Answer a research question grounded exclusively in retrieved passages with source citations."""
    target_ids = request.document_ids or request.paper_ids
    response = container.rag_service.answer(
        question=request.question,
        top_k=request.top_k,
        paper_ids=target_ids,
        mode=request.mode,
        similarity_threshold=request.similarity_threshold,
    )
    container.document_store.log_question(
        question=response.question,
        abstained=response.abstained,
        retrieval_status=response.retrieval_status,
        max_similarity=response.max_similarity_score,
        total_ms=response.latencies_ms.get("total_ms", 0.0),
        asked_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    return response


@router.post("/search", response_model=list[RetrievedPassage])
def search_documents(
    request: SearchRequest,
    container: ServiceContainer = Depends(get_container),
) -> list[RetrievedPassage]:
    """Retrieve the top-K relevant passages across all or selected uploaded papers."""
    target_ids = request.document_ids or request.paper_ids
    return container.search_engine.search(
        query=request.query,
        top_k=request.top_k,
        mode=request.mode,
        paper_ids=target_ids,
        section_types=request.section_types,
    )


@router.get("/documents")
def list_documents(container: ServiceContainer = Depends(get_container)) -> dict:
    """List all indexed documents without exposing internal server file paths."""
    raw_papers = container.document_store.list_papers()
    documents = [
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
        for p in raw_papers
    ]
    return {"count": len(documents), "documents": documents}


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    container: ServiceContainer = Depends(get_container),
) -> dict:
    """Delete a document and remove all of its passages from SQLite and FAISS."""
    deleted = container.document_store.delete_paper(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    removed_vectors = container.vector_store.delete_paper(document_id)
    delete_archived_pdf(document_id, container)
    return {
        "deleted": True,
        "document_id": document_id,
        "vectors_removed": removed_vectors,
    }
