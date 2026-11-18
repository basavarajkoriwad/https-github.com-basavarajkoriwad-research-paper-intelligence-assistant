"""Module 5 Experimental Evaluation Harness for Scientific RAG (Retrieval + Answer Quality + Latency)."""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from scholarmind.api.dependencies import ServiceContainer
from scholarmind.api.routes.papers import ingest_uploaded_pdf


@dataclass
class RAGEvaluationCase:
    """One labeled scientific RAG evaluation case (answerable or unanswerable)."""

    question: str
    expected_doc_name: str | None  # None for unanswerable / out-of-scope questions
    expected_page: int | None
    reference_answer: str
    key_terms: list[str]
    is_answerable: bool = True


def _normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _token_f1(prediction: str, reference: str) -> float:
    pred_toks = _normalize_tokens(prediction)
    ref_toks = _normalize_tokens(reference)
    if not pred_toks or not ref_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(ref_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(ref_toks)
    return 2.0 * precision * recall / (precision + recall)


def evaluate_rag_system(
    container: ServiceContainer,
    documents: list[tuple[str, bytes]],
    eval_cases: list[RAGEvaluationCase],
    top_k: int = 4,
) -> dict[str, Any]:
    """Measure real retrieval, answer quality, citation accuracy, abstention, and performance latencies."""
    # 1. Measure cold ingestion & embedding latency vs. cached re-upload latency
    doc_parse_ms: list[float] = []
    cold_embed_ms: list[float] = []
    cached_embed_ms: list[float] = []

    for filename, pdf_bytes in documents:
        t0 = time.perf_counter()
        parsed, chunks = container.pdf_parser.parse_bytes(pdf_bytes=pdf_bytes, filename=filename)
        doc_parse_ms.append((time.perf_counter() - t0) * 1000.0)

        t_emb0 = time.perf_counter()
        container.document_store.save_paper(parsed, chunks)
        container.vector_store.add_chunks(chunks)
        cold_embed_ms.append((time.perf_counter() - t_emb0) * 1000.0)

        # Measure second upload of identical document (should hit unchanged-doc cache!)
        t_cache0 = time.perf_counter()
        reup = ingest_uploaded_pdf(pdf_bytes=pdf_bytes, filename=filename, container=container)
        cached_embed_ms.append((time.perf_counter() - t_cache0) * 1000.0)
        assert reup.already_indexed is True

    # 2. Evaluate all answerable and unanswerable queries
    precision_at_k: list[float] = []
    recall_at_k: list[float] = []
    mrr_scores: list[float] = []
    answer_f1_scores: list[float] = []
    key_term_coverage: list[float] = []
    evidence_relevance: list[float] = []
    citation_correctness: list[float] = []
    retrieval_latencies: list[float] = []
    generation_latencies: list[float] = []

    unanswerable_total = 0
    unsupported_generated = 0
    per_case_logs: list[dict[str, Any]] = []

    for case in eval_cases:
        res = container.rag_service.answer(question=case.question, top_k=top_k, mode="hybrid")
        retrieval_latencies.append(res.latencies_ms.get("retrieval_ms", 0.0))
        if res.latencies_ms.get("generation_ms", 0.0) > 0:
            generation_latencies.append(res.latencies_ms["generation_ms"])

        if not case.is_answerable:
            unanswerable_total += 1
            if not res.abstained:
                unsupported_generated += 1
            per_case_logs.append(
                {
                    "question": case.question,
                    "is_answerable": False,
                    "abstained": res.abstained,
                    "retrieval_status": res.retrieval_status,
                    "max_similarity_score": res.max_similarity_score,
                }
            )
            continue

        # Retrieval metrics (relevant = matches expected document_name AND expected_page)
        relevant_flags = [
            (p.document_name == case.expected_doc_name and p.page_number == case.expected_page)
            for p in res.retrieved_passages
        ]
        hits = sum(relevant_flags)
        precision_at_k.append(hits / max(1, len(res.retrieved_passages)))
        recall_at_k.append(1.0 if hits > 0 else 0.0)
        first_rank = next((i + 1 for i, rel in enumerate(relevant_flags) if rel), None)
        mrr_scores.append((1.0 / first_rank) if first_rank is not None else 0.0)

        # Evidence relevance: whether top-3 passages contain >= 50% of gold key terms
        top_text = " ".join(p.text.lower() for p in res.retrieved_passages[:3])
        ev_cov = (
            sum(1 for kt in case.key_terms if kt.lower() in top_text) / max(1, len(case.key_terms))
        )
        evidence_relevance.append(ev_cov)

        # Citation correctness: whether returned citations include exact expected doc + page
        cite_match = any(
            s.document_name == case.expected_doc_name and s.page_number == case.expected_page
            for s in res.sources
        )
        citation_correctness.append(1.0 if cite_match else 0.0)

        # Answer correctness (Token-F1 + key-term recall on generated answer body before "Sources:")
        ans_body = res.answer.split("\n\nSources:")[0]
        f1 = _token_f1(ans_body, case.reference_answer)
        kt_hit = (
            sum(1 for kt in case.key_terms if kt.lower() in ans_body.lower()) / max(1, len(case.key_terms))
        )
        answer_f1_scores.append(f1)
        key_term_coverage.append(kt_hit)

        per_case_logs.append(
            {
                "question": case.question,
                "is_answerable": True,
                "abstained": res.abstained,
                "retrieval_status": res.retrieval_status,
                "max_similarity_score": res.max_similarity_score,
                "citation_correct": cite_match,
                "answer_token_f1": round(f1, 4),
                "key_term_recall": round(kt_hit, 4),
                "generated_answer": ans_body,
            }
        )

    def _mean(xs: list[float]) -> float:
        return round(float(np.mean(xs)), 4) if xs else 0.0

    return {
        "embedding_model": container.vector_store.model_name,
        "llm_backend": container.llm_client.backend_name,
        "top_k": top_k,
        "num_documents": len(documents),
        "num_answerable_queries": len(recall_at_k),
        "num_unanswerable_queries": unanswerable_total,
        "retrieval_metrics": {
            f"precision_at_{top_k}": _mean(precision_at_k),
            f"recall_at_{top_k}": _mean(recall_at_k),
            "mrr": _mean(mrr_scores),
        },
        "answer_quality_metrics": {
            "answer_token_f1": _mean(answer_f1_scores),
            "key_term_coverage": _mean(key_term_coverage),
            "evidence_relevance": _mean(evidence_relevance),
            "citation_correctness": _mean(citation_correctness),
            "unsupported_answer_rate": round(
                unsupported_generated / max(1, unanswerable_total), 4
            ),
        },
        "performance_metrics_ms": {
            "avg_doc_processing_ms": round(float(np.mean(doc_parse_ms)), 2),
            "avg_cold_embedding_and_index_ms": round(float(np.mean(cold_embed_ms)), 2),
            "avg_cached_reupload_ms": round(float(np.mean(cached_embed_ms)), 2),
            "avg_retrieval_latency_ms": round(float(np.mean(retrieval_latencies)), 2),
            "avg_answer_generation_latency_ms": round(float(np.mean(generation_latencies)), 2)
            if generation_latencies
            else 0.0,
        },
        "per_case_results": per_case_logs,
    }
