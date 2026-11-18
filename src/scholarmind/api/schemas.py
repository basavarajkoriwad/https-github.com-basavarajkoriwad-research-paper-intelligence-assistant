"""API request and response Pydantic v2 validation schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    indexed_papers: int
    indexed_vectors: int
    llm_backend: str


class PaperUploadResponse(BaseModel):
    document_id: str
    document_name: str
    paper_id: str
    filename: str
    title: str
    authors: str
    page_count: int
    pages_processed: int
    truncated: bool
    word_count: int
    chunks_indexed: int
    already_indexed: bool = False
    sections_detected: list[str]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000, description="Semantic search query.")
    top_k: int = Field(default=5, ge=1, le=25)
    mode: Literal["hybrid", "dense", "lexical"] = "hybrid"
    paper_ids: list[str] | None = None
    document_ids: list[str] | None = None
    section_types: list[str] | None = None

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Search query must not be blank.")
        return v.strip()


class QARequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Research question.")
    top_k: int = Field(default=4, ge=1, le=12)
    mode: Literal["hybrid", "dense", "lexical"] = "hybrid"
    paper_ids: list[str] | None = None
    document_ids: list[str] | None = None
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question must not be blank.")
        return v.strip()


class SummarizeRequest(BaseModel):
    strategy: Literal["auto", "single_pass", "hierarchical"] = "auto"
    max_chunks: int = Field(default=4, ge=1, le=10)


class CompareRequest(BaseModel):
    paper_ids: list[str] = Field(..., min_length=2, max_length=10)

    @field_validator("paper_ids")
    @classmethod
    def unique_paper_ids(cls, v: list[str]) -> list[str]:
        cleaned = [pid.strip() for pid in v if pid and pid.strip()]
        if len(set(cleaned)) < 2:
            raise ValueError("Provide at least 2 distinct paper_ids for comparison.")
        return list(dict.fromkeys(cleaned))
