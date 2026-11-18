"""Scientific PDF ingestion, page-preserving parsing, and SQLite document persistence."""

from scholarmind.ingestion.document_store import SQLiteDocumentStore
from scholarmind.ingestion.pdf_parser import (
    PDFCorruptedError,
    PDFTooLargeError,
    PDFValidationError,
    PyMuPDFParser,
)

__all__ = [
    "PDFCorruptedError",
    "PDFTooLargeError",
    "PDFValidationError",
    "PyMuPDFParser",
    "SQLiteDocumentStore",
]
