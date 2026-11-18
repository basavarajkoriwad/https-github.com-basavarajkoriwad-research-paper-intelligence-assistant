"""SQLite persistence layer for parsed research papers, chunks, and cached analyses."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from scholarmind.core.logging import get_logger
from scholarmind.core.schemas import (
    DocumentChunk,
    ParsedPaper,
    ResearchAnalysisReport,
)

logger = get_logger(__name__)


class SQLiteDocumentStore:
    """Lightweight, ACID-compliant SQLite store for paper metadata, pages, chunks, and analyses."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    title TEXT NOT NULL,
                    authors TEXT,
                    abstract TEXT,
                    page_count INTEGER NOT NULL,
                    pages_processed INTEGER NOT NULL,
                    word_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    ingested_at_utc TEXT NOT NULL,
                    parsed_json TEXT NOT NULL,
                    analysis_json TEXT
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    document_name TEXT NOT NULL DEFAULT '',
                    paper_title TEXT NOT NULL,
                    section_title TEXT NOT NULL,
                    section_type TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    contextualized_text TEXT NOT NULL,
                    word_count INTEGER NOT NULL,
                    estimated_tokens INTEGER NOT NULL,
                    has_math INTEGER NOT NULL,
                    has_citation INTEGER NOT NULL,
                    FOREIGN KEY(paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section_type);

                CREATE TABLE IF NOT EXISTS qa_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    abstained INTEGER NOT NULL,
                    retrieval_status TEXT NOT NULL,
                    max_similarity REAL NOT NULL DEFAULT 0.0,
                    total_ms REAL NOT NULL DEFAULT 0.0,
                    asked_at_utc TEXT NOT NULL
                );
                """
            )
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
            if "document_name" not in cols:
                conn.execute("ALTER TABLE chunks ADD COLUMN document_name TEXT NOT NULL DEFAULT ''")

    def paper_exists(self, paper_id: str) -> bool:
        """Check whether a document with this content hash ID is already stored."""
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        return row is not None

    def save_paper(self, paper: ParsedPaper, chunks: list[DocumentChunk]) -> None:
        """Insert or replace a parsed paper and all its page-tagged chunks atomically."""
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE paper_id = ?", (paper.paper_id,))
            conn.execute(
                """
                INSERT OR REPLACE INTO papers (
                    paper_id, filename, title, authors, abstract,
                    page_count, pages_processed, word_count, chunk_count,
                    ingested_at_utc, parsed_json, analysis_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    paper.paper_id,
                    paper.filename,
                    paper.title,
                    paper.authors,
                    paper.abstract,
                    paper.page_count,
                    paper.pages_processed,
                    paper.word_count,
                    len(chunks),
                    paper.ingested_at_utc,
                    paper.model_dump_json(),
                ),
            )
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, paper_id, chunk_index, document_name, paper_title, section_title,
                    section_type, page_number, text, contextualized_text,
                    word_count, estimated_tokens, has_math, has_citation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.chunk_id,
                        c.paper_id,
                        c.chunk_index,
                        c.document_name or paper.filename,
                        c.paper_title,
                        c.section_title,
                        c.section_type,
                        c.page_number,
                        c.text,
                        c.contextualized_text,
                        c.word_count,
                        c.estimated_tokens,
                        int(c.has_math),
                        int(c.has_citation),
                    )
                    for c in chunks
                ],
            )
        logger.info("Persisted paper '%s' (%d chunks) in %s.", paper.paper_id, len(chunks), self.db_path)

    def list_papers(self) -> list[dict[str, Any]]:
        """Return metadata summaries for all ingested papers."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT paper_id, filename, title, authors, page_count,
                       pages_processed, word_count, chunk_count, ingested_at_utc
                FROM papers
                ORDER BY ingested_at_utc DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def get_paper(self, paper_id: str) -> ParsedPaper | None:
        """Retrieve a full `ParsedPaper` by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT parsed_json FROM papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()
        if not row:
            return None
        return ParsedPaper.model_validate_json(row["parsed_json"])

    def get_chunks(self, paper_id: str | None = None) -> list[DocumentChunk]:
        """Retrieve all stored chunks, optionally filtered to one paper."""
        with self._connect() as conn:
            if paper_id is not None:
                rows = conn.execute(
                    "SELECT * FROM chunks WHERE paper_id = ? ORDER BY chunk_index ASC",
                    (paper_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM chunks ORDER BY paper_id ASC, chunk_index ASC"
                ).fetchall()

        return [
            DocumentChunk(
                chunk_id=r["chunk_id"],
                paper_id=r["paper_id"],
                chunk_index=r["chunk_index"],
                document_name=r["document_name"] if "document_name" in r.keys() else "",
                paper_title=r["paper_title"],
                section_title=r["section_title"],
                section_type=r["section_type"],
                page_number=r["page_number"],
                text=r["text"],
                contextualized_text=r["contextualized_text"],
                word_count=r["word_count"],
                estimated_tokens=r["estimated_tokens"],
                has_math=bool(r["has_math"]),
                has_citation=bool(r["has_citation"]),
            )
            for r in rows
        ]

    def save_analysis(self, report: ResearchAnalysisReport) -> None:
        """Cache a structured `ResearchAnalysisReport` against a paper."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE papers SET analysis_json = ? WHERE paper_id = ?",
                (report.model_dump_json(), report.paper_id),
            )

    def get_analysis(self, paper_id: str) -> ResearchAnalysisReport | None:
        """Return cached `ResearchAnalysisReport` if present."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT analysis_json FROM papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()
        if not row or not row["analysis_json"]:
            return None
        return ResearchAnalysisReport.model_validate(json.loads(row["analysis_json"]))

    def delete_paper(self, paper_id: str) -> bool:
        """Delete a paper and cascade-delete all its chunks."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))
            deleted = cur.rowcount > 0
        if deleted:
            logger.info("Deleted paper '%s' from SQLite store.", paper_id)
        return deleted

    def log_question(
        self,
        question: str,
        abstained: bool,
        retrieval_status: str,
        max_similarity: float,
        total_ms: float,
        asked_at_utc: str,
    ) -> None:
        """Persist a Q&A interaction event for Dashboard telemetry."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO qa_logs (
                    question, abstained, retrieval_status, max_similarity, total_ms, asked_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    question,
                    int(abstained),
                    retrieval_status,
                    float(max_similarity),
                    float(total_ms),
                    asked_at_utc,
                ),
            )

    def get_qa_stats(self, recent_limit: int = 10) -> dict[str, Any]:
        """Return aggregate Q&A telemetry and recent questions for the Dashboard."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS total,
                       COALESCE(SUM(abstained), 0) AS abstained_count,
                       COALESCE(AVG(total_ms), 0.0) AS avg_latency_ms
                FROM qa_logs
                """
            ).fetchone()
            recent_rows = conn.execute(
                """
                SELECT question, abstained, retrieval_status, max_similarity, total_ms, asked_at_utc
                FROM qa_logs
                ORDER BY log_id DESC
                LIMIT ?
                """,
                (recent_limit,),
            ).fetchall()

        total = int(row["total"]) if row else 0
        abstained_count = int(row["abstained_count"]) if row else 0
        return {
            "questions_asked": total,
            "abstained_questions": abstained_count,
            "answered_questions": total - abstained_count,
            "avg_qa_latency_ms": round(float(row["avg_latency_ms"]), 1) if row else 0.0,
            "recent_questions": [
                {
                    "question": r["question"],
                    "abstained": bool(r["abstained"]),
                    "retrieval_status": r["retrieval_status"],
                    "max_similarity": round(float(r["max_similarity"]), 3),
                    "total_ms": round(float(r["total_ms"]), 1),
                    "asked_at_utc": r["asked_at_utc"],
                }
                for r in recent_rows
            ],
        }
