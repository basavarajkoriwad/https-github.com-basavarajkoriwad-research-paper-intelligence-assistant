"""Grounded RAG Question Answering route."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from scholarmind.api.dependencies import ServiceContainer, get_container
from scholarmind.api.schemas import QARequest
from scholarmind.core.schemas import RAGAnswerResponse

router = APIRouter(prefix="/api/v1/qa", tags=["RAG Question Answering"])


@router.post("", response_model=RAGAnswerResponse)
def ask_question(
    request: QARequest,
    container: ServiceContainer = Depends(get_container),
) -> RAGAnswerResponse:
    """Answer a research question grounded in uploaded papers with exact document & page citations."""
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
