"""T5 Long-Document Summarization route."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from scholarmind.api.dependencies import ServiceContainer, get_container
from scholarmind.api.schemas import SummarizeRequest
from scholarmind.core.schemas import SummaryResponse

router = APIRouter(prefix="/api/v1/summarize", tags=["Summarization"])


@router.post("/{paper_id}", response_model=SummaryResponse)
def summarize_paper(
    paper_id: str,
    request: SummarizeRequest | None = None,
    container: ServiceContainer = Depends(get_container),
) -> SummaryResponse:
    """Summarize an uploaded paper using our fine-tuned T5 model (single-pass or chunked Map-Reduce)."""
    req = request or SummarizeRequest()
    paper = container.document_store.get_paper(paper_id)
    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper '{paper_id}' not found.",
        )
    chunks = container.document_store.get_chunks(paper_id=paper_id)
    return container.summarization_service.summarize_paper(
        paper=paper,
        chunks=chunks,
        strategy=req.strategy,
        max_chunks=req.max_chunks,
    )
