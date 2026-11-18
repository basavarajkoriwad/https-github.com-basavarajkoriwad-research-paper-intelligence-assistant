"""Run Module 5 real quantitative RAG evaluation and generate docs/MODULE5_RAG_REPORT.md."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pymupdf as fitz

_ROOT = Path(__file__).resolve().parents[1]
_HF_CACHE = _ROOT / "artifacts" / "hf_cache"
if _HF_CACHE.exists() and not os.getenv("HF_HOME"):
    os.environ["HF_HOME"] = str(_HF_CACHE)
    os.environ["HF_HUB_CACHE"] = str(_HF_CACHE / "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

from scholarmind.api.dependencies import ServiceContainer  # noqa: E402
from scholarmind.core.config import load_config  # noqa: E402
from scholarmind.evaluation.rag_eval import RAGEvaluationCase, evaluate_rag_system  # noqa: E402


def _make_pdf(pages: list[str], title: str, author: str) -> bytes:
    doc = fitz.open()
    doc.set_metadata({"title": title, "author": author})
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 780), text, fontsize=11)
    raw = doc.tobytes()
    doc.close()
    return raw


def build_eval_corpus() -> tuple[list[tuple[str, bytes]], list[RAGEvaluationCase]]:
    paper1 = _make_pdf(
        pages=[
            (
                "Abstract\n"
                "We propose SciLongFormer, a sparse hierarchical Transformer architecture designed to summarize "
                "long scientific articles exceeding 8,000 tokens. The main objective of this research paper is "
                "to preserve methodological accuracy while reducing quadratic self-attention memory overhead.\n\n"
                "1. Datasets and Evaluation\n"
                "We evaluate SciLongFormer on the ArXiv Summarization and PubMed datasets using ROUGE-1, "
                "ROUGE-2, ROUGE-L, and BERTScore evaluation metrics."
            ),
            (
                "2. Methodology\n"
                "The authors selected a hybrid local-global Transformer architecture because standard RNNs and LSTMs "
                "suffer from vanishing gradients across long sequences while dense BERT models truncate at 512 tokens. "
                "We fine-tune the model using LoRA rank-16 adapters and the AdamW optimizer with cosine learning rate decay."
            ),
            (
                "3. Experimental Findings\n"
                "SciLongFormer achieves 46.8 ROUGE-1 and 20.4 ROUGE-2 on ArXiv Summarization, outperforming "
                "traditional BiLSTM baselines by 9.2 ROUGE-1 points while cutting peak GPU memory by 48%.\n\n"
                "4. Limitations and Future Work\n"
                "A key limitation is that SciLongFormer has not been evaluated on non-English clinical reports and "
                "struggles with dense multi-page matrix derivations. For future improvements, the authors suggest "
                "multimodal figure-table parsing and INT8 ONNX quantization for laptop CPUs."
            ),
        ],
        title="SciLongFormer: Sparse Hierarchical Scientific Summarization",
        author="A. Patel, R. Chen",
    )

    paper2 = _make_pdf(
        pages=[
            (
                "Abstract\n"
                "We introduce CiteGuardRAG, a retrieval-augmented generation framework built to reduce hallucinations "
                "in scientific question answering. The system indexes section-aware chunks using all-MiniLM-L6-v2 "
                "embeddings into a FAISS vector index fused with BM25 lexical scoring."
            ),
            (
                "2. Experimental Setup and Findings\n"
                "We benchmark CiteGuardRAG on SciQ and SQuAD 2.0 using Recall@5, Mean Reciprocal Rank (MRR), "
                "and Exact Match metrics. CiteGuardRAG achieves 0.94 MRR on SciQ and reduces unsupported answers "
                "by 67% compared to ungrounded LLM generation.\n\n"
                "3. Limitations\n"
                "CiteGuardRAG cannot process password-protected or image-only scanned PDFs without an external OCR engine."
            ),
        ],
        title="CiteGuardRAG: Grounded Scientific Question Answering",
        author="M. Kumar, S. Lopez",
    )

    cases = [
        RAGEvaluationCase(
            question="What is the main objective of the SciLongFormer research paper?",
            expected_doc_name="SciLongFormer_2025.pdf",
            expected_page=1,
            reference_answer="To summarize long scientific articles while preserving methodological accuracy and reducing quadratic self-attention memory overhead.",
            key_terms=["summarize", "memory", "self-attention"],
            is_answerable=True,
        ),
        RAGEvaluationCase(
            question="Which datasets and evaluation metrics were used to evaluate SciLongFormer?",
            expected_doc_name="SciLongFormer_2025.pdf",
            expected_page=1,
            reference_answer="ArXiv Summarization and PubMed datasets evaluated with ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore.",
            key_terms=["arxiv", "pubmed", "rouge"],
            is_answerable=True,
        ),
        RAGEvaluationCase(
            question="Why did the authors select a hybrid local-global Transformer architecture in SciLongFormer?",
            expected_doc_name="SciLongFormer_2025.pdf",
            expected_page=2,
            reference_answer="Because RNNs and LSTMs suffer from vanishing gradients across long sequences and dense BERT truncates at 512 tokens.",
            key_terms=["vanishing", "512", "rnn"],
            is_answerable=True,
        ),
        RAGEvaluationCase(
            question="What were the experimental findings of SciLongFormer on ArXiv Summarization?",
            expected_doc_name="SciLongFormer_2025.pdf",
            expected_page=3,
            reference_answer="It achieves 46.8 ROUGE-1 and 20.4 ROUGE-2, outperforming BiLSTM baselines by 9.2 points while reducing GPU memory by 48%.",
            key_terms=["46.8", "bilstm", "48%"],
            is_answerable=True,
        ),
        RAGEvaluationCase(
            question="What future improvements do the SciLongFormer authors suggest?",
            expected_doc_name="SciLongFormer_2025.pdf",
            expected_page=3,
            reference_answer="Multimodal figure-table parsing and INT8 ONNX quantization for laptop CPUs.",
            key_terms=["multimodal", "onnx", "quantization"],
            is_answerable=True,
        ),
        RAGEvaluationCase(
            question="What embeddings and retrieval components does CiteGuardRAG use?",
            expected_doc_name="CiteGuardRAG_2025.pdf",
            expected_page=1,
            reference_answer="all-MiniLM-L6-v2 embeddings indexed in FAISS fused with BM25 lexical scoring.",
            key_terms=["all-minilm", "faiss", "bm25"],
            is_answerable=True,
        ),
        RAGEvaluationCase(
            question="What are the limitations of CiteGuardRAG?",
            expected_doc_name="CiteGuardRAG_2025.pdf",
            expected_page=2,
            reference_answer="CiteGuardRAG cannot process password-protected or image-only scanned PDFs without an external OCR engine.",
            key_terms=["scanned", "ocr", "password"],
            is_answerable=True,
        ),
        # Unanswerable / out-of-scope questions testing abstention guardrail
        RAGEvaluationCase(
            question="What is the traditional recipe for Neapolitan margherita pizza dough?",
            expected_doc_name=None,
            expected_page=None,
            reference_answer="",
            key_terms=[],
            is_answerable=False,
        ),
        RAGEvaluationCase(
            question="Which football team won the 2022 FIFA World Cup final?",
            expected_doc_name=None,
            expected_page=None,
            reference_answer="",
            key_terms=[],
            is_answerable=False,
        ),
    ]

    return [("SciLongFormer_2025.pdf", paper1), ("CiteGuardRAG_2025.pdf", paper2)], cases


def main() -> None:
    cfg = load_config()
    container = ServiceContainer(config=cfg, store_dir=Path(tempfile.mkdtemp()))
    docs, cases = build_eval_corpus()
    results = evaluate_rag_system(container=container, documents=docs, eval_cases=cases, top_k=4)

    out_json = cfg.artifacts_dir / "experiments" / "module5_rag_eval.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ret = results["retrieval_metrics"]
    ans = results["answer_quality_metrics"]
    perf = results["performance_metrics_ms"]

    k = results["top_k"]
    p_at_k = ret[f"precision_at_{k}"]
    r_at_k = ret[f"recall_at_{k}"]

    md_lines = [
        "# Module 5: Research Paper RAG Evaluation Report",
        "",
        f"- **Embedding Model:** `{results['embedding_model']}` (384-dim, cosine-normalized, SHA-256 cached)",
        "- **Vector Index:** `FAISS IndexFlatIP` + `BM25Okapi` (RRF Hybrid)",
        f"- **Open-Source LLM Backend:** `{results['llm_backend']}`",
        f"- **Evaluation Corpus:** {results['num_documents']} multi-page scientific PDFs, "
        f"{results['num_answerable_queries']} answerable research queries, "
        f"{results['num_unanswerable_queries']} out-of-scope unanswerable queries.",
        "",
        "## 1. Retrieval Metrics (Measured)",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Precision@{k} | {p_at_k:.4f} |",
        f"| Recall@{k} | {r_at_k:.4f} |",
        f"| Mean Reciprocal Rank (MRR) | {ret['mrr']:.4f} |",
        "",
        "## 2. Answer Quality & Grounding Metrics (Measured)",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Answer Token-F1 vs Reference | {ans['answer_token_f1']:.4f} |",
        f"| Key Technical Term Coverage | {ans['key_term_coverage']:.4f} |",
        f"| Retrieved Evidence Relevance | {ans['evidence_relevance']:.4f} |",
        f"| Exact Citation Correctness (Doc + Page) | {ans['citation_correctness']:.4f} |",
        f"| Unsupported Answer Rate (Lower = Better) | {ans['unsupported_answer_rate']:.4f} |",
        "",
        "## 3. CPU Performance & Caching Latencies (Measured)",
        "",
        "| Pipeline Stage | Mean Latency (ms) |",
        "| :--- | ---: |",
        f"| PDF Parsing & Page Chunking | {perf['avg_doc_processing_ms']} ms |",
        f"| Cold Embedding & FAISS Indexing | {perf['avg_cold_embedding_and_index_ms']} ms |",
        f"| Re-Upload of Unchanged PDF (Dedup Skip) | {perf['avg_cached_reupload_ms']} ms |",
        f"| Hybrid Retrieval Latency (FAISS + BM25) | {perf['avg_retrieval_latency_ms']} ms |",
        f"| LLM Answer Generation Latency (`flan-t5-base` CPU) | {perf['avg_answer_generation_latency_ms']} ms |",
        "",
        "## 4. Per-Query Breakdown",
        "",
        "| Question | Answerable | Abstained | Retrieval Status | Top Cosine Sim | Cite Correct | Token-F1 |",
        "| :--- | :---: | :---: | :--- | ---: | :---: | ---: |",
    ]

    for row in results["per_case_results"]:
        cite_str = "Yes" if row.get("citation_correct") else ("N/A" if not row["is_answerable"] else "No")
        f1_str = f"{row['answer_token_f1']:.3f}" if "answer_token_f1" in row else "N/A"
        md_lines.append(
            f"| {row['question'][:60]} | {row['is_answerable']} | {row['abstained']} | "
            f"`{row['retrieval_status']}` | {row['max_similarity_score']:.3f} | {cite_str} | {f1_str} |"
        )

    out_md = cfg.project_root / "docs" / "MODULE5_RAG_REPORT.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved JSON -> {out_json}")
    print(f"Saved Report -> {out_md}")
    print(json.dumps({k: v for k, v in results.items() if k != "per_case_results"}, indent=2))


if __name__ == "__main__":
    main()
