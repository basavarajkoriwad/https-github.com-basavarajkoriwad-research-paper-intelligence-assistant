# Data Acquisition, EDA & NLP Preprocessing Design Document
## Research Paper Intelligence Assistant (`ScholarMind AI`)

This document explains **every data engineering and NLP preprocessing decision** implemented in Stage 2, grounded in scientific NLP best practices and empirical requirements of downstream models (`RNN/BiLSTM`, `BERT/SciBERT/DeBERTa-v3`, `T5/LongT5`, `Embeddings`, and `RAG + LLMs`).

---

## 1. Dataset Acquisition & Scaling Strategy

### Why We Avoid Massive Automatic Downloads
Two of the four reference corpora are multi-gigabyte archives:
- **`Cornell-University/arxiv`** (~3+ GB JSONL metadata; >1 TB raw PDFs; requires Kaggle authentication)
- **`ccdv/arxiv-summarization`** (~7.6 GB full corpus across 203K long articles)

Downloading 10+ GB eagerly slows onboarding, risks disk/RAM exhaustion on 16 GB laptops, and blocks rapid iteration. Therefore, `configs/base.yaml` defines four reproducible scale tiers:

| Scale | ArXiv Metadata | SQuAD 2.0 (Tr/Val) | ArXiv Summ (Tr/Val/Te) | SciQ (Tr/Val/Te) | Typical Use Case |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **`mini`** | 500 | 500 / 200 | 100 / 30 / 30 | 500 / 100 / 100 | Rapid local validation, CI, laptop EDA |
| **`small`** | 5,000 | 3,000 / 800 | 500 / 100 / 100 | 2,000 / 500 / 500 | Laptop CPU development & baseline training |
| **`medium`** | 50,000 | 25,000 / 5,000 | 5,000 / 500 / 500 | Full (11.7K / 1K / 1K) | Consumer GPU fine-tuning & ablations |
| **`full`** | 200,000+ | Full (130K / 11.9K) | Full (203K / 6.4K / 6.4K) | Full (11.7K / 1K / 1K) | Research GPU training runs |

### Manual Kaggle ArXiv Support + Streaming JSONL
- **`src/scholarmind/data/arxiv_loader.py`** streams `arxiv-metadata-oai-snapshot.json` **line-by-line** ($O(1)$ memory overhead per line) instead of `json.load()` on the entire 3+ GB file.
- If the manual Kaggle download is not yet present at `data/raw/arxiv/arxiv-metadata-oai-snapshot.json`, `scripts/download_datasets.py` prints exact manual download instructions AND supports `--arxiv-source api` to fetch a small, real ArXiv sample via the official public `export.arxiv.org` Atom API in the **exact same Kaggle-compatible JSONL schema**.
- Hugging Face loaders (`squad_loader.py`, `arxiv_summ_loader.py`, `sciq_loader.py`) default to **`streaming=True`** so `mini`/`small` runs download only a few megabytes.

---

## 2. Task-Aware Text Cleaning & Normalization (`src/scholarmind/nlp/cleaning.py`)

A common pitfall in NLP pipelines is applying a single aggressive cleaner (lowercasing + punctuation stripping + stopword removal) to every model. That damages modern Transformers and corrupts extractive QA character offsets. Instead, we enforce **four explicit `CleaningProfile` modes**:

| Cleaning Step | Why It Is Necessary | `QA_PRESERVE_OFFSETS` | `SUMMARIZATION_SCIENTIFIC` | `RAG_SEMANTIC` | `CLASSIFICATION_CLASSICAL` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Unicode NFKC & Ligature Repair (`ﬁ`→`fi`)** | PDFs embed typographic ligatures and smart quotes that cause OOV splits and broken keyword matches. | Yes (realigned) | Yes | Yes | Yes |
| **Control Character Removal** | Removes hidden null bytes / PDF extraction artifacts that break tokenizers and Parquet/JSON readers. | Yes | Yes | Yes | Yes |
| **PDF De-Hyphenation (`trans-\nformer`→`transformer`)** | Academic multi-column layouts frequently hyphenate words across line wraps. | No (keeps spans intact) | Yes | Yes | Yes |
| **ArXiv Marker Normalization (`@xmath0`→`[MATH]`, `@xcite`→`[CITE]`, `@xref`→`[REF]`)** | `ccdv/arxiv-summarization` replaces equations/citations with numbered placeholders (`@xmath0`..`@xmath142`), inflating vocabulary sparsity without semantic value. | No | Yes | Yes | Yes |
| **Offset Realignment (`clean_qa_example`)** | Expanding ligatures (`ﬁ`→`fi`) or trimming leading whitespace shifts SQuAD 2.0 `answer_start` character indices. We realign and strictly assert `context[start:start+len(ans)] == ans`, dropping any corrupted span. | **Yes (Strict)** | N/A | N/A | N/A |
| **Lowercasing & Citation Stripping** | Reduces vocabulary sparsity for classical TF-IDF / BM25 / from-scratch RNN embeddings, while **preserved** for BERT/T5/LLMs where casing differentiates acronyms (`BERT`, `US`, `BLEU`, `ReLU`) from ordinary words. | No | No | No | **Yes** |

---

## 3. Scientific-Aware Sentence Segmentation (`src/scholarmind/nlp/segmentation.py`)

Standard sentence splitters break scientific prose on periods inside citations and notation:
- `"Vaswani et al. introduced..."` → wrongly splits after `al.`
- `"As shown in Fig. 2 and Sec. 3.1, accuracy rose to 92.4."` → wrongly splits after `Fig.`, `Sec.`, or `3.`

**Our Solution:** `ScientificSentenceSegmenter` masks 30+ academic abbreviations (`et al.`, `Fig.`, `Eq.`, `Sec.`, `e.g.`, `i.e.`, `vs.`, `w.r.t.`, `cf.`, `Ref.`, `Vol.`, `pp.`) and numeric decimals (`(?<=\d)\.(?=\d)`) with a `<DOT>` sentinel before invoking **spaCy** or **NLTK (`punkt_tab`)**, then restores periods cleanly. This directly protects downstream **sentence-aligned scientific chunking**.

---

## 4. Tokenization & Train-Only Vocabulary (`src/scholarmind/nlp/tokenization.py`)

- **`ScientificTokenizer`** supports three backends (`scientific_regex`, `nltk`, `spacy`). The default `scientific_regex` backend explicitly keeps domain constructs intact (`self-attention`, `DeBERTa-v3`, `F1-score`, `1e-4`, `[MATH]`, `[CITE]`) instead of fragmenting them on hyphens or brackets.
- **`Vocabulary`** (used by Phase 2 PyTorch RNN/BiLSTM models) reserves `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, filters rare noisy tokens (`min_freq=2`), and is **fitted exclusively on the `train` split** (`data/processed/arxiv_metadata/rnn_vocab.json`) to prevent vocabulary leakage from validation or test papers.
- **`estimate_subword_tokens()`** uses a calibrated `1.30x` scientific word-to-subword ratio for rapid CPU chunk sizing and EDA exceedance checks (`>512` and `>4096` tokens) without requiring heavy Transformer tokenizer weights during preprocessing.

---

## 5. Stopword Handling & Lemmatization Policy (`src/scholarmind/nlp/normalization.py`)

### Where It IS Applied vs. Where It Is NOT Applied
Enforced explicitly via `should_apply_classical_normalization(task)`:
- **Applied (`True`):** Lexical search (`BM25`), `TF-IDF`, EDA keyword frequency analysis, topic modeling, and optional classical baseline features (`classical_text` column).
- **NOT Applied (`False`):** `BERT/SciBERT/DeBERTa` Extractive QA, `T5/LongT5` Summarization, Dense Semantic Embeddings (`BGE-M3`), and `RAG + LLM` prompts — contextual Transformers rely on prepositions, verb tense, and syntactic structure.

### Mandatory Negation & Discourse Cue Preservation
Standard English stopword lists (including NLTK's) delete `"not"`, `"no"`, `"nor"`, `"without"`, `"but"`, `"however"`, `"only"`, `"more"`, `"less"`. Removing these in scientific papers is **catastrophic**:
- `"Method A does not outperform Method B"` → `"method outperform method"` (inverted claim!)
- `"However, the model fails without large pretraining"` → loses limitation/research-gap signals!

Therefore, `PRESERVED_NEGATIONS_AND_CUES` explicitly subtracts 40+ negations and contrast/limitation cues from the stopword set.

### Lemmatization Backend Strategy
Prefers **spaCy (`en_core_web_sm`)** or **NLTK `WordNetLemmatizer` with POS-tag mapping** (converting Penn Treebank `V/J/R/N` tags to WordNet POS so `"trained"` → `"train"` and `"better"` → `"good"`), with a conservative rule-based fallback that refuses to mangle acronyms or hyphenated terms.

---

## 6. Scientific Document Chunking (`src/scholarmind/nlp/chunking.py`)

Empirical EDA on `ccdv/arxiv-summarization` confirms that **>95% of scientific articles exceed BERT's 512-token limit** and a substantial fraction exceed 4,096 tokens. Naive fixed-character splitting severs sentences, mixes `Methodology` with `Limitations`, and loses paper/section context.

We implement **`HierarchicalSectionChunker`** (alongside a `FixedWindowChunker` ablation baseline):
1. **Section Detection (`ScientificSectionDetector`):** Parses Markdown, numbered (`3.2 Experimental Setup`), and bare academic headers into canonical section types (`abstract`, `introduction`, `related_work`, `methodology`, `experiments`, `results`, `limitations`, `discussion`, `conclusion`, `references`).
2. **Parent-Child Hierarchy:**
   - **Parent Chunks** (up to ~1,500 tokens): Preserve broad section narrative for LLM context expansion.
   - **Child Chunks** (up to 384 estimated tokens): Precision retrieval units packed strictly along **scientific sentence boundaries** with **2-sentence overlap** to preserve cross-sentence references.
3. **Contextual Breadcrumbs:** Every chunk stores both raw `text` and `contextualized_text` prefixed with `[Paper: <title> | Section: <section_title>]`, ensuring dense embeddings retain document & section provenance.
4. **Scientific Metadata Flags:** Records `has_math`, `has_citation`, `section_type`, `word_count`, `estimated_tokens`, and `sentence_count` for filtered hybrid search (e.g., querying `section_type == "limitations"` during Stage-1 Research Gap Discovery).

---

## 7. Leakage Prevention & Splitting Strategy (`src/scholarmind/data/splits.py`)

Data leakage silently inflates research metrics and undermines scientific validity. Our pipeline enforces **four layers of protection**:

1. **Pre-Split Deduplication:** Exact IDs and normalized SHA-1 content hashes (`compute_text_hash`) are deduplicated *before* splitting.
2. **Group-Aware Splitting (`group_aware_split`):**
   - **ArXiv Metadata:** Grouped by normalized `title_group` hash and stratified by `primary_category` (`80/10/10`) so revised versions or duplicate titles of the same paper never cross train/val/test.
   - **SQuAD 2.0:** Official validation split is held out cleanly as `test`; official `train` is split `85/15` into `train`/`val` **grouped by Wikipedia article `title_group`** so the QA model never sees questions from the same article in both train and validation.
   - **ArXiv Summarization & Chunks:** Official `train`/`val`/`test` splits are preserved and **chunked strictly after split isolation**, ensuring chunks from the same `paper_id` never leak across splits.
   - **SciQ:** Official `train`/`val`/`test` splits are audited by `support_group` passage hash and `question` hash so identical supporting passages do not leak from train into val/test.
3. **Cross-Split Leakage Removal (`remove_cross_split_leakage`):** When official splits contain accidental duplicates, evaluation splits (`test`, then `val`) are preserved untouched and overlapping records are pruned from `train`.
4. **Automated Leakage Audit (`verify_no_leakage`):** Asserts `0` ID overlaps, `0` group overlaps, and `0` text-hash overlaps across every split pair, and writes the signed `LeakageAuditReport` directly into each dataset's `manifest.json`.

---

## 8. Processed Storage Format (`src/scholarmind/data/storage.py`)

All processed splits are persisted as **Snappy-compressed Apache Parquet (`.parquet`)** accompanied by a checksummed **`manifest.json`**:
- **Why Parquet over CSV/JSON:** Columnar binary format preserves Python types (`list[str]`, `bool`, `int64`), handles embedded newlines/LaTeX/quotes safely without CSV escaping corruption, loads 5–10x faster into Pandas & Hugging Face `datasets`, and reduces disk footprint by ~60–75%.
- **Reproducibility:** Every `manifest.json` records dataset source, scale, random seed, preprocessing profile, split row counts, SHA-256 file checksums, and the `LeakageAuditReport`.
