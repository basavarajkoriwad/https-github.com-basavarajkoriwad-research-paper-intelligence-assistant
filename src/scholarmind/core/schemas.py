"""Canonical Pydantic v2 schemas shared across data, NLP, ingestion, retrieval, and API layers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ParsedSection(BaseModel):
    """Represents a logical section extracted from a scientific document."""

    section_id: str
    paper_id: str
    section_title: str
    section_type: str  # abstract, introduction, methodology, results, limitations, conclusion, etc.
    order: int
    text: str
    page_number: int = 1
    char_start: int = 0
    char_end: int = 0


class PageContent(BaseModel):
    """Per-page extracted text preserving PDF page provenance."""

    page_number: int
    text: str
    word_count: int = 0
    char_count: int = 0


class ParsedPaper(BaseModel):
    """Canonical representation of an ingested scientific PDF document."""

    paper_id: str
    filename: str
    title: str
    authors: str = ""
    abstract: str = ""
    page_count: int
    pages_processed: int
    truncated: bool = False
    word_count: int = 0
    ingested_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    pages: list[PageContent] = Field(default_factory=list)
    sections: list[ParsedSection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Structured scientific text chunk for embeddings, retrieval, and provenance."""

    chunk_id: str
    paper_id: str
    parent_id: str | None = None
    chunk_level: str = "child"  # "parent" | "child"
    document_name: str = ""
    paper_title: str = ""
    section_title: str = "Body"
    section_type: str = "body"
    page_number: int = 1
    chunk_index: int = 0
    text: str
    contextualized_text: str
    char_start: int = 0
    char_end: int = 0
    word_count: int = 0
    estimated_tokens: int = 0
    sentence_count: int = 0
    has_math: bool = False
    has_citation: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedPassage(BaseModel):
    """A ranked passage returned by semantic or hybrid search with full provenance."""

    rank: int = 1
    chunk_id: str
    paper_id: str
    document_name: str = ""
    paper_title: str
    section_title: str
    section_type: str
    page_number: int
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0
    text: str


class CitationReference(BaseModel):
    """Explicit grounded source reference linking a claim to document and page."""

    rank: int = 1
    paper_id: str
    document_name: str = ""
    paper_title: str
    page_number: int | None = None
    section_title: str
    chunk_id: str
    similarity_score: float = 0.0
    quote: str


class RAGAnswerResponse(BaseModel):
    """Structured RAG Q&A response separating retrieval similarity from answer reliability."""

    question: str
    answer: str
    grounded: bool
    abstained: bool
    retrieval_status: str = "insufficient_evidence"
    similarity_scores: list[float] = Field(default_factory=list)
    max_similarity_score: float = 0.0
    evidence_note: str = (
        "Cosine similarity reflects semantic retrieval proximity, not factual correctness probability."
    )
    llm_backend: str
    sources: list[CitationReference] = Field(default_factory=list)
    citations: list[CitationReference] = Field(default_factory=list)
    retrieved_passages: list[RetrievedPassage] = Field(default_factory=list)
    latencies_ms: dict[str, float] = Field(default_factory=dict)


class SummaryResponse(BaseModel):
    """Structured T5 summarization response."""

    paper_id: str
    paper_title: str
    strategy: str
    model_name: str
    used_finetuned_checkpoint: bool
    page_count: int
    chunks_processed: int
    summary: str


class GroundedFindingItem(BaseModel):
    """An extracted scientific statement tied to its source page and section."""

    statement: str
    page_number: int
    section_title: str
    source_type: str = "author_stated"  # "author_stated" | "derived_from_limitation"


class ResearchAnalysisReport(BaseModel):
    """Grounded structured research intelligence extracted from a paper."""

    paper_id: str
    paper_title: str
    objectives: list[GroundedFindingItem] = Field(default_factory=list)
    methodologies: list[GroundedFindingItem] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    model_architectures: list[str] = Field(default_factory=list)
    key_findings: list[GroundedFindingItem] = Field(default_factory=list)
    explicit_limitations: list[GroundedFindingItem] = Field(default_factory=list)
    future_directions: list[GroundedFindingItem] = Field(default_factory=list)
    extraction_notes: list[str] = Field(default_factory=list)


class PaperComparisonEntry(BaseModel):
    """Per-paper structured column inside a multi-paper comparison matrix."""

    paper_id: str
    paper_title: str
    page_count: int
    primary_objective: str
    methodology_summary: str
    datasets: list[str] = Field(default_factory=list)
    model_architectures: list[str] = Field(default_factory=list)
    reported_results: list[str] = Field(default_factory=list)
    explicit_limitations: list[str] = Field(default_factory=list)


class PaperComparisonResponse(BaseModel):
    """Structured comparison across 2 or more research papers."""

    paper_ids: list[str]
    comparison_matrix: list[PaperComparisonEntry]
    shared_datasets: list[str] = Field(default_factory=list)
    shared_models: list[str] = Field(default_factory=list)
    comparative_summary: str
    citations: list[CitationReference] = Field(default_factory=list)


class LeakageAuditReport(BaseModel):
    """Audit result proving zero data leakage across dataset splits."""

    dataset_name: str
    passed: bool
    id_overlaps: dict[str, int] = Field(default_factory=dict)
    group_overlaps: dict[str, int] = Field(default_factory=dict)
    text_hash_overlaps: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class SplitManifest(BaseModel):
    """Reproducible metadata manifest stored alongside processed Parquet splits."""

    dataset_name: str
    source: str
    scale: str
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    seed: int = 42
    preprocessing_profile: str
    split_counts: dict[str, int]
    columns: list[str]
    files: dict[str, str]
    sha256_checksums: dict[str, str] = Field(default_factory=dict)
    leakage_audit: LeakageAuditReport
    notes: dict[str, Any] = Field(default_factory=dict)
