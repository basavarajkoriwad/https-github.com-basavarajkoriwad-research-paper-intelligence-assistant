# Phase 1: Complete Project Deployment Audit Report
## Research Paper Intelligence Assistant (`ScholarMind AI`)

**Audit Date:** 2026-11-18  
**Scope:** Full workspace inspection (`90` Python files, `5` Jupyter notebooks, `4` YAML configs, `30` automated tests, saved checkpoints & runtime artifacts) prior to GitHub CI/CD and Microsoft Azure deployment.

---

## 1. Workspace & Component Inventory

| Category | Verified Modules / Files | Status |
| :--- | :--- | :--- |
| **Core & Config** | `src/scholarmind/core/` (`config.py`, `device.py`, `logging.py`, `schemas.py`), `configs/base.yaml`, `configs/profiles/*.yaml` | Working |
| **Datasets & EDA** | `src/scholarmind/data/` (`arxiv_loader.py`, `squad_loader.py`, `arxiv_summ_loader.py`, `sciq_loader.py`, `eda.py`, `splits.py`, `storage.py`, `pipeline.py`) | Working |
| **Scientific NLP** | `src/scholarmind/nlp/` (`cleaning.py`, `segmentation.py`, `tokenization.py`, `normalization.py`, `chunking.py`) | Working |
| **Deep Learning (PyTorch)** | `src/scholarmind/models/rnn_lstm/` (Vanilla RNN, LSTM, BiLSTM+Attention), `models/bert/` (SciBERT), `models/t5/` (Flan-T5), `models/embeddings/` (TF-IDF, SciBERT, SentenceTransformer), `models/llm/` (Flan-T5-Base + Ollama) | Working (Pure PyTorch; zero TensorFlow dependency) |
| **PDF & Persistence** | `src/scholarmind/ingestion/` (`pdf_parser.py` via PyMuPDF, `document_store.py` via SQLite) | Working |
| **Vector DB & Hybrid RAG** | `src/scholarmind/retrieval/` (`vector_store.py` FAISS CPU, `embedding_cache.py`, `hybrid_search.py` FAISS+BM25 RRF), `src/scholarmind/services/` (`rag_qa.py`, `summarization.py`, `research_analysis.py`, `comparison.py`) | Working |
| **FastAPI Backend** | `src/scholarmind/api/` (`main.py`, `dependencies.py`, `schemas.py`, `routes/{module5,papers,search,qa,summarize,analysis,system}.py`) · Entry point: `scripts/run_api.py` / `uvicorn scholarmind.api.main:app` | Working (Verified live HTTP) |
| **Streamlit Frontend** | `src/scholarmind/ui/` (`app.py` 10-section workbench, `api_client.py`) · Entry point: `scripts/run_ui.py` / `streamlit run src/scholarmind/ui/app.py` | Working (Verified live HTTP) |
| **Evaluation & Tests** | `tests/unit/*`, `tests/integration/*` (30 tests passing), `scripts/run_experiments.py`, `scripts/evaluate_rag_module5.py` | Working |

---

## 2. Artifact Size & GitHub Repository Limit Audit

| Directory / File | Measured Size | GitHub / Cloud Strategy |
| :--- | ---: | :--- |
| `.venv/` | 1,530.1 MB | **Gitignored** (rebuilt from `requirements.txt` / Docker) |
| `artifacts/hf_cache/` | ~4,000 MB | **Gitignored** (downloaded on demand into Azure persistent volume / optional build cache) |
| `artifacts/checkpoints/exp_c_bert/best_model.pt` | **419.41 MB** | **Exceeds GitHub 100 MB limit — Gitignored** (store in Azure Blob / File Share if needed) |
| `artifacts/checkpoints/exp_d_t5/best_model.pt` | **293.67 MB** | **Exceeds GitHub 100 MB limit — Gitignored** (app gracefully falls back to `google/flan-t5-small` or loads mounted checkpoint from Azure File Share) |
| `artifacts/checkpoints/exp_a/b/e` | 15.08 MB | **Gitignored** (reproducible via `scripts/run_experiments.py`) |
| `artifacts/experiments/*.json` | < 0.1 MB | **Tracked in Git** (powers cloud Dashboard Experiments & Evaluation tabs after path sanitization) |
| `src/`, `tests/`, `scripts/`, `docs/`, `notebooks/` | ~2.0 MB | **Tracked in Git** |

---

## 3. Issues, Gaps & Deployment Risks Identified (Remediated in Phases 2–6)

| # | Finding | Severity | Remediation Applied |
| :--- | :--- | :--- | :--- |
| **1** | **Local Machine Paths in Reports/JSONs:** 17 occurrences of `C:\Users\<local-user>\...` found in `artifacts/experiments/*.json` and `docs/EXPERIMENTS_REPORT.md`. | High (Privacy / Portability) | Sanitize all saved JSON/Markdown paths to portable repo-relative POSIX paths and update experiment writers. |
| **2** | **`.agents/` Directory Not in `.gitignore`:** Internal IDE directory present in workspace root. | Medium | Add `.agents/` and `*.log` to `.gitignore`. |
| **3** | **Filename Sanitization & Safe PDF Archiving:** Uploaded PDF filenames need strict path-traversal stripping (`Path(filename).name`) and optional safe raw PDF storage under `data/store/uploads/` with synchronized delete. | Medium | Harden `ingest_uploaded_pdf` and `delete_paper` / `delete_document`. |
| **4** | **Dev vs. Production Config Separation:** CORS defaulted to `allow_origins=["*"]` with no `production.yaml` or production 500 error masking. | Medium | Add `configs/profiles/production.yaml`, `SCHOLARMIND_ENV`, `SCHOLARMIND_CORS_ORIGINS`, and sanitized 500 exception handler. |
| **5** | **Heavy Models in CI Tests:** Full integration tests load ~1.5 GB HF models; running them directly in GitHub Actions on every PR would slow CI and hammer HF Hub. | High (CI Reliability) | Mark HF-heavy tests with `@pytest.mark.slow_model` and add a fast, thorough `tests/ci/test_ci_api_smoke.py` testing every API endpoint + FAISS + SQLite + RAG abstention/citations in <15s without downloading multi-GB LLMs. |
| **6** | **Containerization & Cloud Inter-Service URL:** No `Dockerfile`, `.dockerignore`, or `docker-compose.yml` yet; Streamlit must talk to FastAPI via configurable `SCHOLARMIND_API_URL` (never hardcoded `localhost`). | High (Deployment Blocker) | Build unified CPU-optimized `Dockerfile`, `.dockerignore`, `docker/entrypoint.sh`, and `docker-compose.yml` (`SCHOLARMIND_API_URL=http://api:8000`). |
| **7** | **GitHub Repository & Local CLI Check:** `git` and `docker` are installed; `gh` and `az` are not installed locally. API query to `github.com/basavarajkoriwad` found **0 existing public repositories** and no local `.git` repo initialized yet. | Info | Initialize clean local Git repo, prepare GitHub Actions CI + Azure Container Apps CD workflows, and request user approval before first push. |
| **8** | **Secrets Scan Result:** **Zero** hardcoded API keys, passwords, or `.env` files found in `src/`, `scripts/`, `configs/`, or `tests/`. | Verified Safe | Maintain strict `.gitignore` rules protecting `.env`, `kaggle.json`, `*.key`, `*.pem`. |

---

## 4. Required Cloud Infrastructure (Microsoft Azure)

- **Recommended Service:** **Azure Container Apps (ACA)** (selected over Azure App Service for Containers for native dual-service networking between FastAPI + Streamlit, built-in Azure Files persistent mounts for SQLite/FAISS/HF cache, and flexible CPU/RAM scaling).
- **Recommended Production Resources:**
  - **Azure Container Registry (ACR)** — Basic/Standard tier
  - **Azure Storage Account (Azure Files)** — 2 persistent shares (`scholarmind-data` → `/app/data`, `scholarmind-hf-cache` → `/app/artifacts/hf_cache`)
  - **Azure Container Apps Environment** with 2 apps:
    - `scholarmind-api` (2 vCPU, 4 GiB RAM, internal/external ingress `:8000`)
    - `scholarmind-ui` (1 vCPU, 2 GiB RAM, external HTTPS ingress `:8501`, `SCHOLARMIND_API_URL=https://<api-fqdn>`)
