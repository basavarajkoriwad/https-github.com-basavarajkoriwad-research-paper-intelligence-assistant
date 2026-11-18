# Module 5: Research Paper RAG Evaluation Report

- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine-normalized, SHA-256 cached)
- **Vector Index:** `FAISS IndexFlatIP` + `BM25Okapi` (RRF Hybrid)
- **Open-Source LLM Backend:** `local-pytorch:google/flan-t5-base`
- **Evaluation Corpus:** 2 multi-page scientific PDFs, 7 answerable research queries, 2 out-of-scope unanswerable queries.

## 1. Retrieval Metrics (Measured)

| Metric | Value |
| :--- | ---: |
| Precision@4 | 0.4286 |
| Recall@4 | 1.0000 |
| Mean Reciprocal Rank (MRR) | 0.9286 |

## 2. Answer Quality & Grounding Metrics (Measured)

| Metric | Value |
| :--- | ---: |
| Answer Token-F1 vs Reference | 0.7535 |
| Key Technical Term Coverage | 0.8095 |
| Retrieved Evidence Relevance | 1.0000 |
| Exact Citation Correctness (Doc + Page) | 1.0000 |
| Unsupported Answer Rate (Lower = Better) | 0.0000 |

## 3. CPU Performance & Caching Latencies (Measured)

| Pipeline Stage | Mean Latency (ms) |
| :--- | ---: |
| PDF Parsing & Page Chunking | 15.35 ms |
| Cold Embedding & FAISS Indexing | 2978.04 ms |
| Re-Upload of Unchanged PDF (Dedup Skip) | 4.86 ms |
| Hybrid Retrieval Latency (FAISS + BM25) | 16.13 ms |
| LLM Answer Generation Latency (`flan-t5-base` CPU) | 2249.56 ms |

## 4. Per-Query Breakdown

| Question | Answerable | Abstained | Retrieval Status | Top Cosine Sim | Cite Correct | Token-F1 |
| :--- | :---: | :---: | :--- | ---: | :---: | ---: |
| What is the main objective of the SciLongFormer research pap | True | False | `weak_evidence` | 0.460 | Yes | 0.692 |
| Which datasets and evaluation metrics were used to evaluate  | True | False | `moderate_evidence` | 0.613 | Yes | 0.421 |
| Why did the authors select a hybrid local-global Transformer | True | False | `moderate_evidence` | 0.486 | Yes | 0.842 |
| What were the experimental findings of SciLongFormer on ArXi | True | False | `strong_evidence` | 0.778 | Yes | 0.764 |
| What future improvements do the SciLongFormer authors sugges | True | False | `moderate_evidence` | 0.558 | Yes | 1.000 |
| What embeddings and retrieval components does CiteGuardRAG u | True | False | `moderate_evidence` | 0.563 | Yes | 0.556 |
| What are the limitations of CiteGuardRAG? | True | False | `moderate_evidence` | 0.558 | Yes | 1.000 |
| What is the traditional recipe for Neapolitan margherita piz | False | True | `insufficient_evidence` | 0.139 | N/A | N/A |
| Which football team won the 2022 FIFA World Cup final? | False | True | `insufficient_evidence` | 0.095 | N/A | N/A |
