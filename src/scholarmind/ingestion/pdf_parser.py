"""Robust scientific PDF parser preserving page numbers, metadata, and section chunks."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from scholarmind.core.logging import get_logger
from scholarmind.core.schemas import (
    DocumentChunk,
    PageContent,
    ParsedPaper,
    ParsedSection,
)
from scholarmind.nlp.chunking import (
    HierarchicalSectionChunker,
    ScientificSectionDetector,
)
from scholarmind.nlp.cleaning import CleaningProfile, ScientificTextCleaner

logger = get_logger(__name__)


class PDFValidationError(ValueError):
    """Raised when an uploaded file fails basic PDF validation checks."""


class PDFTooLargeError(PDFValidationError):
    """Raised when an uploaded PDF exceeds the configured maximum byte size."""


class PDFCorruptedError(PDFValidationError):
    """Raised when a PDF is corrupted, encrypted, or yields no extractable text."""


class PyMuPDFParser:
    """Page-preserving scientific PDF parser handling large and corrupted files safely.

    Reuses `ScientificTextCleaner`, `ScientificSectionDetector`, and
    `HierarchicalSectionChunker` from `scholarmind.nlp` without duplicating NLP logic.
    """

    def __init__(
        self,
        max_file_size_mb: float = 50.0,
        max_pages: int = 150,
        max_tokens_per_chunk: int = 384,
        overlap_sentences: int = 2,
    ) -> None:
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.max_pages = max_pages
        self.cleaner = ScientificTextCleaner(profile=CleaningProfile.RAG_SEMANTIC)
        self.section_detector = ScientificSectionDetector()
        self.chunker = HierarchicalSectionChunker(
            max_tokens=max_tokens_per_chunk,
            overlap_sentences=overlap_sentences,
            include_breadcrumb=True,
        )

    @staticmethod
    def _infer_title(pdf_meta: dict, first_page_text: str, filename: str) -> str:
        meta_title = str(pdf_meta.get("title") or "").strip()
        if meta_title and len(meta_title) >= 5 and not meta_title.lower().startswith("untitled"):
            return re.sub(r"\s+", " ", meta_title)[:220]

        # Heuristic fallback: first non-empty meaningful line on page 1
        for line in first_page_text.splitlines():
            clean_line = line.strip()
            if (
                len(clean_line) >= 8
                and not clean_line.lower().startswith(("arxiv:", "preprint", "published as", "proceedings"))
                and not clean_line.isdigit()
            ):
                return clean_line[:220]

        return Path(filename).stem.replace("_", " ").replace("-", " ").strip() or "Untitled Paper"

    @staticmethod
    def _extract_abstract(sections: list[ParsedSection], pages: list[PageContent]) -> str:
        for sec in sections:
            if sec.section_type == "abstract" and len(sec.text.split()) >= 15:
                return sec.text[:2500]
        if pages:
            m = re.search(
                r"(?:^|\n)abstract\s*[:.\-]?\s*\n?([\s\S]{80,2200}?)(?:\n\s*(?:1\.?\s+introduction|introduction|keywords|index terms))",
                pages[0].text,
                re.IGNORECASE,
            )
            if m:
                return m.group(1).strip()
            return " ".join(pages[0].text.split()[:180])
        return ""

    def parse_bytes(
        self,
        pdf_bytes: bytes,
        filename: str,
        paper_id: str | None = None,
    ) -> tuple[ParsedPaper, list[DocumentChunk]]:
        """Parse raw PDF bytes into a `ParsedPaper` and page-tagged `DocumentChunk` list."""
        if not pdf_bytes:
            raise PDFValidationError("Uploaded PDF file is empty.")

        if len(pdf_bytes) > self.max_file_size_bytes:
            size_mb = len(pdf_bytes) / (1024 * 1024)
            limit_mb = self.max_file_size_bytes / (1024 * 1024)
            raise PDFTooLargeError(
                f"PDF file size ({size_mb:.2f} MB) exceeds maximum limit ({limit_mb:.1f} MB)."
            )

        if not pdf_bytes[:1024].lstrip().startswith(b"%PDF-"):
            raise PDFCorruptedError("Invalid PDF header: file does not start with '%PDF-'.")

        computed_id = paper_id or f"paper_{hashlib.sha256(pdf_bytes).hexdigest()[:12]}"

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            raise PDFCorruptedError(f"Unable to open PDF (file may be corrupted): {exc}") from exc

        try:
            if doc.is_encrypted:
                raise PDFCorruptedError("PDF is password-protected/encrypted and cannot be parsed.")

            total_pages = int(doc.page_count)
            if total_pages == 0:
                raise PDFCorruptedError("PDF contains zero pages.")

            pages_to_read = min(total_pages, self.max_pages)
            truncated = total_pages > self.max_pages
            pdf_meta = dict(doc.metadata or {})

            pages: list[PageContent] = []
            all_sections: list[ParsedSection] = []
            all_chunks: list[DocumentChunk] = []

            active_section_title = "Front Matter"
            active_section_type = "body"
            sec_order = 0

            # First pass: extract cleaned per-page text preserving exact 1-based page numbers
            for page_idx in range(pages_to_read):
                page = doc.load_page(page_idx)
                # `sort=True` improves reading order on multi-column academic pages
                raw_text = page.get_text("text", sort=True)
                clean_text = self.cleaner.clean(raw_text)
                pages.append(
                    PageContent(
                        page_number=page_idx + 1,
                        text=clean_text,
                        word_count=len(clean_text.split()),
                        char_count=len(clean_text),
                    )
                )

            non_empty_words = sum(p.word_count for p in pages)
            if non_empty_words < 10:
                raise PDFCorruptedError(
                    "PDF yielded virtually no extractable text (file may be image-only/scanned or damaged)."
                )

            title = self._infer_title(pdf_meta, pages[0].text if pages else "", filename)
            authors = str(pdf_meta.get("author") or "").strip()

            # Second pass: section detection + hierarchical chunking per page so EVERY chunk
            # preserves exact document_name, page_number, and section_type provenance without duplicates!
            global_chunk_idx = 0
            seen_chunk_hashes: set[str] = set()
            for page_obj in pages:
                if not page_obj.text.strip():
                    continue
                page_sections = self.section_detector.split_into_sections(
                    text=page_obj.text,
                    paper_id=computed_id,
                    default_title=active_section_title,
                )

                for sec in page_sections:
                    # Carry over active section title/type across page breaks
                    if sec.section_title != active_section_title:
                        active_section_title = sec.section_title
                        active_section_type = sec.section_type

                    sec_with_page = ParsedSection(
                        section_id=f"{computed_id}::p{page_obj.page_number:03d}_s{sec_order:03d}",
                        paper_id=computed_id,
                        section_title=sec.section_title,
                        section_type=sec.section_type if sec.section_type != "body" else active_section_type,
                        order=sec_order,
                        text=sec.text,
                        page_number=page_obj.page_number,
                        char_start=sec.char_start,
                        char_end=sec.char_end,
                    )
                    all_sections.append(sec_with_page)
                    sec_order += 1

                    page_chunks = self.chunker.chunk_document(
                        text=sec_with_page.text,
                        paper_id=computed_id,
                        paper_title=title,
                        sections=[sec_with_page],
                        include_parents=False,
                        extra_metadata={
                            "page_number": page_obj.page_number,
                            "document_name": filename,
                        },
                    )
                    for ch in page_chunks:
                        norm_text = re.sub(r"\s+", " ", ch.text).strip()
                        if not norm_text:
                            continue
                        chunk_sig = hashlib.sha1(
                            f"{page_obj.page_number}:{norm_text.lower()}".encode("utf-8")
                        ).hexdigest()
                        if chunk_sig in seen_chunk_hashes:
                            continue
                        seen_chunk_hashes.add(chunk_sig)

                        ch.document_name = filename
                        ch.page_number = page_obj.page_number
                        ch.chunk_index = global_chunk_idx
                        ch.chunk_id = f"{computed_id}::p{page_obj.page_number:03d}_c{global_chunk_idx:04d}"
                        ch.contextualized_text = (
                            f"[Document: {filename} | Paper: {title[:90]} | "
                            f"Section: {ch.section_title} | Page: {ch.page_number}]\n"
                            f"{ch.text}"
                        )
                        all_chunks.append(ch)
                        global_chunk_idx += 1

            abstract = self._extract_abstract(all_sections, pages)
            parsed = ParsedPaper(
                paper_id=computed_id,
                filename=filename,
                title=title,
                authors=authors,
                abstract=abstract,
                page_count=total_pages,
                pages_processed=pages_to_read,
                truncated=truncated,
                word_count=non_empty_words,
                pages=pages,
                sections=all_sections,
                metadata={
                    "pdf_subject": str(pdf_meta.get("subject") or ""),
                    "pdf_creator": str(pdf_meta.get("creator") or ""),
                    "pdf_creation_date": str(pdf_meta.get("creationDate") or ""),
                    "file_size_bytes": len(pdf_bytes),
                    "chunk_count": len(all_chunks),
                },
            )
            logger.info(
                "Parsed PDF '%s' (id=%s) -> %d/%d pages, %d words, %d sections, %d chunks.",
                filename,
                computed_id,
                pages_to_read,
                total_pages,
                non_empty_words,
                len(all_sections),
                len(all_chunks),
            )
            return parsed, all_chunks
        finally:
            doc.close()

    def parse_file(self, path: Path, paper_id: str | None = None) -> tuple[ParsedPaper, list[DocumentChunk]]:
        """Read a PDF file from disk and parse it."""
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        return self.parse_bytes(path.read_bytes(), filename=path.name, paper_id=paper_id)
