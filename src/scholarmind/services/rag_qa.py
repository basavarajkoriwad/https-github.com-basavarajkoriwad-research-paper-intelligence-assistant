"""Module 5: Advanced Grounded RAG Q&A Engine with structured prompt, verification & similarity status.

Key engineering guarantees:
1. Separates retrieval similarity (cosine score) from answer reliability (`retrieval_status` + `evidence_note`).
   Never labels cosine similarity as factual confidence.
2. Deduplicates near-identical retrieved passages and filters weak/irrelevant noise before LLM generation.
3. Uses a strict grounded research prompt enforcing zero fabrication and explicit abstention.
4. Builds all citations strictly from verified FAISS/SQLite chunk metadata (`document_name`, `page_number`).
"""

from __future__ import annotations

import time
from typing import Literal

from scholarmind.core.logging import get_logger
from scholarmind.core.schemas import (
    CitationReference,
    RAGAnswerResponse,
    RetrievedPassage,
)
from scholarmind.models.llm.open_llm import BaseLLMClient
from scholarmind.nlp.normalization import ScientificNormalizer
from scholarmind.retrieval.hybrid_search import HybridSearchEngine

logger = get_logger(__name__)

ABSTENTION_MESSAGE = (
    "The uploaded research documents do not provide sufficient information to answer this question reliably. "
    "No unsupported answer was generated."
)

SIMILARITY_DISCLAIMER = (
    "Cosine similarity measures retrieval match between the question and passages, "
    "not the probability that an answer is factually correct."
)


class RAGQAService:
    """Grounded Retrieval-Augmented Generation service for scientific research papers."""

    def __init__(
        self,
        search_engine: HybridSearchEngine,
        llm_client: BaseLLMClient,
        min_similarity_threshold: float = 0.34,
        moderate_similarity_threshold: float = 0.48,
        strong_similarity_threshold: float = 0.62,
        min_term_overlap: float = 0.20,
    ) -> None:
        self.search_engine = search_engine
        self.llm_client = llm_client
        self.min_similarity_threshold = min_similarity_threshold
        self.moderate_similarity_threshold = moderate_similarity_threshold
        self.strong_similarity_threshold = strong_similarity_threshold
        self.min_term_overlap = min_term_overlap
        self.normalizer = ScientificNormalizer(backend="auto", preserve_negations=True)

    def _compute_term_overlap(self, question: str, passages: list[RetrievedPassage]) -> float:
        q_terms = set(self.normalizer.normalize_text(question, remove_stopwords=True, lemmatize=False))
        if not q_terms:
            return 0.0
        best_overlap = 0.0
        for p in passages:
            p_terms = set(self.normalizer.normalize_text(p.text, remove_stopwords=True, lemmatize=False))
            overlap = len(q_terms & p_terms) / len(q_terms)
            if overlap > best_overlap:
                best_overlap = overlap
        return best_overlap

    @staticmethod
    def _deduplicate_passages(
        passages: list[RetrievedPassage],
        jaccard_threshold: float = 0.85,
    ) -> list[RetrievedPassage]:
        """Remove exact or near-duplicate passages while preserving rank order."""
        kept: list[RetrievedPassage] = []
        kept_token_sets: list[set[str]] = []

        for p in passages:
            tokens = set(p.text.lower().split())
            if not tokens:
                continue
            is_dup = False
            for prev_tokens in kept_token_sets:
                union_len = len(tokens | prev_tokens)
                if union_len > 0 and (len(tokens & prev_tokens) / union_len) >= jaccard_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(p.model_copy(update={"rank": len(kept) + 1}))
                kept_token_sets.append(tokens)
        return kept

    def _classify_retrieval_status(
        self,
        max_dense: float,
        term_overlap: float,
        min_threshold: float,
    ) -> tuple[Literal["strong_evidence", "moderate_evidence", "weak_evidence", "insufficient_evidence"], str]:
        if max_dense < min_threshold and not (term_overlap >= self.min_term_overlap and max_dense >= min_threshold * 0.75):
            return (
                "insufficient_evidence",
                "Retrieved passages fell below the relevance threshold; abstained to avoid unsupported claims. "
                + SIMILARITY_DISCLAIMER,
            )
        if max_dense >= self.strong_similarity_threshold:
            return (
                "strong_evidence",
                "Strong retrieval match found in uploaded papers. " + SIMILARITY_DISCLAIMER,
            )
        if max_dense >= self.moderate_similarity_threshold:
            return (
                "moderate_evidence",
                "Moderate retrieval match found; verify answer against the cited supporting passages. "
                + SIMILARITY_DISCLAIMER,
            )
        return (
            "weak_evidence",
            "Weak retrieval similarity detected; treat the generated explanation cautiously and inspect the source passages. "
            + SIMILARITY_DISCLAIMER,
        )

    def _build_grounded_prompt(self, question: str, passages: list[RetrievedPassage]) -> str:
        """Construct structured research-assistant RAG prompt tailored to open-source LLMs."""
        context_blocks: list[str] = []
        for idx, p in enumerate(passages, start=1):
            doc_label = p.document_name or p.paper_title
            page_label = f"Page {p.page_number}" if p.page_number else "Page unavailable"
            snippet = " ".join(p.text.split()[:120])
            context_blocks.append(
                f"[{idx}] Source: {doc_label} ({page_label}, Section: {p.section_title})\n"
                f"{snippet}"
            )
        retrieved_context = "\n\n".join(context_blocks)

        # Ollama / instruction chat models accept long multi-rule system prompts cleanly;
        # Flan-T5 has a 512-768 token encoder window so we keep all grounding rules concise.
        if self.llm_client.backend_name.startswith("ollama:"):
            return (
                "SYSTEM ROLE:\n"
                "You are an AI research assistant specializing in scientific literature analysis. "
                "Your responsibility is to answer questions using only the supplied research document context.\n\n"
                "RULES:\n"
                "1. Use only the retrieved research passages as evidence.\n"
                "2. Do not invent facts, results, methodologies or citations.\n"
                "3. Do not use unsupported external knowledge to complete missing answers.\n"
                "4. If the context does not contain sufficient information, explicitly state that the uploaded "
                "documents do not provide sufficient information.\n"
                "5. Distinguish between the authors' findings and your own explanation.\n"
                "6. Preserve technical terminology and explain complex concepts clearly.\n"
                "7. Cite the source document and page number for important claims.\n"
                "8. If different papers provide conflicting information, explain the differences.\n"
                "9. Never generate a citation that does not correspond to an actual retrieved passage.\n\n"
                f"USER QUESTION:\n{question}\n\n"
                f"RETRIEVED RESEARCH CONTEXT:\n{retrieved_context}\n\n"
                "INSTRUCTIONS:\n"
                "Generate a clear, structured answer based exclusively on the retrieved context and cite source documents and page numbers."
            )

        return (
            "You are a scientific research assistant. Answer the question using ONLY the retrieved "
            "research passages below. Do not invent facts, numbers, or citations. If the passages do not "
            "contain enough information, reply exactly: 'The uploaded documents do not provide sufficient information.'\n\n"
            f"RETRIEVED RESEARCH CONTEXT:\n{retrieved_context}\n\n"
            f"USER QUESTION: {question}\n\n"
            "GROUNDED ANSWER:"
        )

    def answer(
        self,
        question: str,
        top_k: int = 4,
        paper_ids: list[str] | None = None,
        mode: str = "hybrid",
        similarity_threshold: float | None = None,
    ) -> RAGAnswerResponse:
        """Execute full grounded RAG: retrieve -> deduplicate -> gate -> generate -> verify citations."""
        t_total0 = time.perf_counter()
        clean_q = (question or "").strip()
        if not clean_q:
            return RAGAnswerResponse(
                question="",
                answer="Please provide a non-empty research question.",
                grounded=True,
                abstained=True,
                retrieval_status="insufficient_evidence",
                similarity_scores=[],
                max_similarity_score=0.0,
                evidence_note="Empty question submitted.",
                llm_backend=self.llm_client.backend_name,
            )

        min_thresh = (
            float(similarity_threshold)
            if similarity_threshold is not None
            else self.min_similarity_threshold
        )

        # 1. Retrieve top-K passages (over-fetch slightly to allow near-duplicate removal)
        t_ret0 = time.perf_counter()
        raw_passages = self.search_engine.search(
            query=clean_q,
            top_k=max(top_k + 2, 5),
            mode=mode,  # type: ignore[arg-type]
            paper_ids=paper_ids,
        )
        passages = self._deduplicate_passages(raw_passages)[:top_k]
        retrieval_ms = round((time.perf_counter() - t_ret0) * 1000.0, 2)

        if not passages:
            return RAGAnswerResponse(
                question=clean_q,
                answer=ABSTENTION_MESSAGE,
                grounded=True,
                abstained=True,
                retrieval_status="insufficient_evidence",
                similarity_scores=[],
                max_similarity_score=0.0,
                evidence_note="No documents are currently indexed or matched the selected filter.",
                llm_backend=self.llm_client.backend_name,
                sources=[],
                citations=[],
                retrieved_passages=[],
                latencies_ms={
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": 0.0,
                    "total_ms": round((time.perf_counter() - t_total0) * 1000.0, 2),
                },
            )

        # Cosine similarity scores from dense FAISS search
        sim_scores = [round(float(p.dense_score), 4) for p in passages]
        max_dense = max(sim_scores, default=0.0)
        term_overlap = self._compute_term_overlap(clean_q, passages)

        # Lexical-only mode guardrail check
        if mode == "lexical":
            max_lex = max((p.lexical_score for p in passages), default=0.0)
            if max_lex <= 0.5 or term_overlap < self.min_term_overlap:
                status_label, evidence_note = "insufficient_evidence", (
                    "Insufficient lexical evidence matched in uploaded documents. " + SIMILARITY_DISCLAIMER
                )
            else:
                status_label, evidence_note = "moderate_evidence", (
                    "Lexical BM25 evidence retrieved. " + SIMILARITY_DISCLAIMER
                )
        else:
            status_label, evidence_note = self._classify_retrieval_status(
                max_dense=max_dense,
                term_overlap=term_overlap,
                min_threshold=min_thresh,
            )

        if status_label == "insufficient_evidence":
            logger.info(
                "RAG abstained on query='%s' (max_dense=%.3f, thresh=%.3f, overlap=%.3f).",
                clean_q[:80],
                max_dense,
                min_thresh,
                term_overlap,
            )
            return RAGAnswerResponse(
                question=clean_q,
                answer=ABSTENTION_MESSAGE,
                grounded=True,
                abstained=True,
                retrieval_status=status_label,
                similarity_scores=sim_scores,
                max_similarity_score=round(max_dense, 4),
                evidence_note=evidence_note,
                llm_backend=self.llm_client.backend_name,
                sources=[],
                citations=[],
                retrieved_passages=passages,
                latencies_ms={
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": 0.0,
                    "total_ms": round((time.perf_counter() - t_total0) * 1000.0, 2),
                },
            )

        # 2. Keep passages that meet a reasonable floor relative to top hit (avoid noisy tail)
        evidence_passages = [
            p for p in passages[: min(3, len(passages))]
            if mode == "lexical" or p.dense_score >= max(0.20, min_thresh * 0.70)
        ] or passages[:1]

        # 3. Generate grounded answer via open-source LLM
        prompt = self._build_grounded_prompt(clean_q, evidence_passages)
        t_gen0 = time.perf_counter()
        raw_answer = self.llm_client.generate(prompt, max_new_tokens=180).strip()
        generation_ms = round((time.perf_counter() - t_gen0) * 1000.0, 2)

        low_ans = raw_answer.lower()
        if (
            not raw_answer
            or "do not provide sufficient information" in low_ans
            or "not stated in the provided documents" in low_ans
            or "insufficient evidence" in low_ans
        ):
            return RAGAnswerResponse(
                question=clean_q,
                answer=ABSTENTION_MESSAGE,
                grounded=True,
                abstained=True,
                retrieval_status="insufficient_evidence",
                similarity_scores=sim_scores,
                max_similarity_score=round(max_dense, 4),
                evidence_note=(
                    "The LLM determined that retrieved passages did not sufficiently answer the question. "
                    + SIMILARITY_DISCLAIMER
                ),
                llm_backend=self.llm_client.backend_name,
                sources=[],
                citations=[],
                retrieved_passages=passages,
                latencies_ms={
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": generation_ms,
                    "total_ms": round((time.perf_counter() - t_total0) * 1000.0, 2),
                },
            )

        # 4. Build verified citations strictly from retrieved metadata (never LLM-invented!)
        verified_sources = [
            CitationReference(
                rank=idx,
                paper_id=p.paper_id,
                document_name=p.document_name or p.paper_title,
                paper_title=p.paper_title,
                page_number=p.page_number if p.page_number and p.page_number > 0 else None,
                section_title=p.section_title,
                chunk_id=p.chunk_id,
                similarity_score=round(float(p.dense_score), 4),
                quote=" ".join(p.text.split()[:60]),
            )
            for idx, p in enumerate(evidence_passages, start=1)
        ]

        source_lines = []
        for s in verified_sources:
            pg = f"Page {s.page_number}" if s.page_number is not None else "Page unavailable"
            source_lines.append(f"[{s.rank}] {s.document_name} ({pg}, {s.section_title})")
        formatted_answer = f"{raw_answer}\n\nSources: " + "; ".join(source_lines) + "."

        return RAGAnswerResponse(
            question=clean_q,
            answer=formatted_answer,
            grounded=True,
            abstained=False,
            retrieval_status=status_label,
            similarity_scores=sim_scores,
            max_similarity_score=round(max_dense, 4),
            evidence_note=evidence_note,
            llm_backend=self.llm_client.backend_name,
            sources=verified_sources,
            citations=verified_sources,
            retrieved_passages=passages,
            latencies_ms={
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": round((time.perf_counter() - t_total0) * 1000.0, 2),
            },
        )
