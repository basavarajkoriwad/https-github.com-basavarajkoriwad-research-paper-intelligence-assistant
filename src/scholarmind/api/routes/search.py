"""Semantic and hybrid search route."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from scholarmind.api.dependencies import ServiceContainer, get_container
from scholarmind.api.schemas import SearchRequest
from scholarmind.core.schemas import RetrievedPassage

router = APIRouter(prefix="/api/v1/search", tags=["Semantic Search"])


@router.post("", response_model=list[RetrievedPassage])
def semantic_search(
    request: SearchRequest,
    container: ServiceContainer = Depends(get_container),
) -> list[RetrievedPassage]:
    """Perform dense, lexical, or RRF-hybrid semantic search over indexed research passages."""
    target_ids = request.document_ids or request.paper_ids
    return container.search_engine.search(
        query=request.query,
        top_k=request.top_k,
        mode=request.mode,
        paper_ids=target_ids,
        section_types=request.section_types,
    )
