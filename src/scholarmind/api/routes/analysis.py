"""Grounded Research Analysis and Multi-Paper Comparison routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from scholarmind.api.dependencies import ServiceContainer, get_container
from scholarmind.api.schemas import CompareRequest
from scholarmind.core.schemas import PaperComparisonResponse, ResearchAnalysisReport

router = APIRouter(prefix="/api/v1", tags=["Research Analysis & Comparison"])


@router.get("/analysis/{paper_id}", response_model=ResearchAnalysisReport)
def analyze_paper(
    paper_id: str,
    refresh: bool = Query(default=False, description="Force re-computation instead of cached report."),
    container: ServiceContainer = Depends(get_container),
) -> ResearchAnalysisReport:
    """Extract objectives, methodologies, datasets, models, findings, limitations & future work."""
    paper = container.document_store.get_paper(paper_id)
    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper '{paper_id}' not found.",
        )

    if not refresh:
        cached = container.document_store.get_analysis(paper_id)
        if cached is not None:
            return cached

    report = container.analysis_service.analyze(paper)
    container.document_store.save_analysis(report)
    return report


@router.post("/compare", response_model=PaperComparisonResponse)
def compare_papers(
    request: CompareRequest,
    container: ServiceContainer = Depends(get_container),
) -> PaperComparisonResponse:
    """Compare 2 to 10 uploaded papers across methods, datasets, models, results, and limitations."""
    papers = []
    analyses = []
    missing: list[str] = []

    for pid in request.paper_ids:
        paper = container.document_store.get_paper(pid)
        if paper is None:
            missing.append(pid)
            continue
        analysis = container.document_store.get_analysis(pid)
        if analysis is None:
            analysis = container.analysis_service.analyze(paper)
            container.document_store.save_analysis(analysis)
        papers.append(paper)
        analyses.append(analysis)

    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Papers not found: {', '.join(missing)}",
        )

    return container.comparison_service.compare(papers=papers, analyses=analyses)
