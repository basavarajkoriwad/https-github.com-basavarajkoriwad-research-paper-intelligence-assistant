"""Dependency container wiring ingestion, SQLite store, FAISS retrieval, and research services."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from scholarmind.core.config import AppConfig, load_config
from scholarmind.ingestion.document_store import SQLiteDocumentStore
from scholarmind.ingestion.pdf_parser import PyMuPDFParser
from scholarmind.models.embeddings.retrieval_benchmark import BaseTextEmbedder
from scholarmind.models.llm.open_llm import BaseLLMClient, create_open_llm_client
from scholarmind.models.t5.t5_summarizer import T5ScientificSummarizer
from scholarmind.retrieval.hybrid_search import HybridSearchEngine
from scholarmind.retrieval.vector_store import FAISSVectorStore
from scholarmind.services.comparison import PaperComparisonService
from scholarmind.services.rag_qa import RAGQAService
from scholarmind.services.research_analysis import ResearchAnalysisService
from scholarmind.services.summarization import PaperSummarizationService


class ServiceContainer:
    """Centralized, testable dependency container for all backend services."""

    def __init__(
        self,
        config: AppConfig | None = None,
        store_dir: Path | None = None,
        embedder: BaseTextEmbedder | None = None,
        llm_client: BaseLLMClient | None = None,
        summarizer: T5ScientificSummarizer | None = None,
    ) -> None:
        self.config = config or load_config()
        self.config.ensure_directories()

        base_store = store_dir or (self.config.data_dir / "store")
        base_store.mkdir(parents=True, exist_ok=True)
        hf_cache = self.config.artifacts_dir / "hf_cache"
        t5_ckpt = self.config.artifacts_dir / "checkpoints" / "exp_d_t5" / "best_model.pt"

        self.pdf_parser = PyMuPDFParser(
            max_file_size_mb=50.0,
            max_pages=150,
            max_tokens_per_chunk=self.config.chunking.max_tokens,
            overlap_sentences=self.config.chunking.overlap_sentences,
        )
        self.document_store = SQLiteDocumentStore(db_path=base_store / "scholarmind.db")
        self.vector_store = FAISSVectorStore(
            index_dir=base_store / "faiss_index",
            embedder=embedder,
            model_name=self.config.models.get(
                "embedding", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            hf_cache_dir=hf_cache,
            enable_embedding_cache=True,
        )
        self.search_engine = HybridSearchEngine(vector_store=self.vector_store)
        rag_flan_model = self.config.models.get("rag_llm", "google/flan-t5-base")
        self.llm_client = llm_client or create_open_llm_client(
            preferred_provider=None,
            ollama_model=self.config.models.get("llm_model", "qwen2.5:3b"),
            flan_model=rag_flan_model,
            hf_cache_dir=hf_cache,
        )
        self.rag_service = RAGQAService(
            search_engine=self.search_engine,
            llm_client=self.llm_client,
        )
        self.summarization_service = PaperSummarizationService(
            model_name=self.config.models.get("summarizer", "google/flan-t5-small"),
            finetuned_checkpoint=t5_ckpt,
            hf_cache_dir=hf_cache,
            summarizer=summarizer,
        )
        self.analysis_service = ResearchAnalysisService()
        self.comparison_service = PaperComparisonService()


@lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    """Return the default singleton `ServiceContainer` (overridable in tests)."""
    return ServiceContainer()
