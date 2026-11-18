"""ScholarMind AI — Complete 10-Section Streamlit Research Paper Intelligence Dashboard.

Sections:
 1. Dashboard
 2. Upload Research Papers
 3. Paper Summarization
 4. Ask Research Assistant
 5. Semantic Search
 6. Research Gap Analysis
 7. Paper Comparison
 8. Deep Learning Experiments
 9. Model Evaluation
10. About Project
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from scholarmind.core.config import load_config
from scholarmind.ui.api_client import ScholarMindClient

SECTIONS: list[str] = [
    "1. Dashboard",
    "2. Upload Research Papers",
    "3. Paper Summarization",
    "4. Ask Research Assistant",
    "5. Semantic Search",
    "6. Research Gap Analysis",
    "7. Paper Comparison",
    "8. Deep Learning Experiments",
    "9. Model Evaluation",
    "10. About Project",
]

QUICK_RESEARCH_QUESTIONS: list[str] = [
    "What is the main objective of this research paper?",
    "Which dataset was used?",
    "Why did the authors select this architecture?",
    "Explain the proposed methodology.",
    "What evaluation metrics were used?",
    "What were the experimental findings?",
    "What are the limitations of this research?",
    "What future improvements do the authors suggest?",
]

STATUS_BADGES: dict[str, tuple[str, str]] = {
    "strong_evidence": ("Strong Retrieval Match", "green"),
    "moderate_evidence": ("Moderate Retrieval Match", "blue"),
    "weak_evidence": ("Weak Retrieval Match — Verify Carefully", "orange"),
    "insufficient_evidence": ("Insufficient Evidence (Abstained)", "red"),
}

SECTION_TYPES: list[str] = [
    "abstract",
    "introduction",
    "related_work",
    "methodology",
    "experiments",
    "results",
    "limitations",
    "discussion",
    "conclusion",
    "body",
]


@st.cache_resource(show_spinner="Connecting to ScholarMind AI Research Engine...")
def _get_client() -> ScholarMindClient:
    return ScholarMindClient()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; }
        .sm-card {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.85rem;
            background-color: rgba(128, 128, 128, 0.05);
        }
        .sm-pill {
            display: inline-block;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(37, 99, 235, 0.15);
            margin-right: 0.35rem;
            margin-bottom: 0.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _plots_dir() -> Path:
    return load_config().reports_dir / "plots"


def _doc_selector(docs: list[dict[str, Any]], label: str, key: str) -> dict[str, Any] | None:
    if not docs:
        st.info("No papers uploaded yet. Go to **2. Upload Research Papers** to upload a PDF.")
        return None
    options = {f"{d['document_name']} — {d['title'][:65]}": d for d in docs}
    selected_label = st.selectbox(label, options=list(options.keys()), key=key)
    return options[selected_label]


# ---------------------------------------------------------------------------
# 1. DASHBOARD
# ---------------------------------------------------------------------------
def render_dashboard(client: ScholarMindClient) -> None:
    st.header("Research Intelligence Dashboard")
    status = client.get_system_status()
    qa_stats = status.get("qa_stats", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Uploaded Papers", status.get("indexed_papers", 0))
    c2.metric("FAISS Indexed Vectors", status.get("indexed_vectors", 0))
    c3.metric("Questions Asked", qa_stats.get("questions_asked", 0))
    c4.metric("Compute Device", status.get("compute_device", "cpu").upper())

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("Uploaded Document Processing Status")
        papers = status.get("papers", [])
        if papers:
            df_docs = pd.DataFrame(papers)[
                ["document_name", "title", "page_count", "word_count", "chunk_count", "ingested_at_utc"]
            ]
            df_docs.columns = ["Filename", "Title", "Pages", "Words", "Indexed Chunks", "Ingested (UTC)"]
            st.dataframe(df_docs, use_container_width=True, hide_index=True)
        else:
            st.info("No research papers indexed yet. Use **2. Upload Research Papers** to ingest your first PDF.")

        st.subheader("Recent Questions Asked")
        recent_q = qa_stats.get("recent_questions", [])
        if recent_q:
            df_q = pd.DataFrame(recent_q)
            df_q.columns = ["Question", "Abstained", "Retrieval Status", "Top Cosine Sim", "Latency (ms)", "Time (UTC)"]
            st.dataframe(df_q, use_container_width=True, hide_index=True)
        else:
            st.caption("No questions recorded yet. Ask a question in **4. Ask Research Assistant**.")

    with col_right:
        st.subheader("Model & Checkpoint Availability")
        st.caption(
            "Explicitly distinguishes models **trained/fine-tuned in PyTorch** from **pretrained inference models**."
        )
        inv = status.get("models_inventory", [])
        if inv:
            df_inv = pd.DataFrame(inv)
            df_inv["Status"] = df_inv["available"].map({True: "Ready", False: "Not Found"})
            df_inv = df_inv[["component", "model_id", "training_status", "Status"]]
            df_inv.columns = ["Component", "Model", "Training Status", "Availability"]
            st.dataframe(df_inv, use_container_width=True, hide_index=True)

        st.markdown(
            f"""
            <div class="sm-card">
                <strong>Active Runtime Configuration</strong><br/>
                • Backend Mode: <code>{client.connection_label}</code><br/>
                • Hardware Profile: <code>{status.get('profile', 'cpu_lite')}</code><br/>
                • RAG Embeddings: <code>{status.get('embedding_model')}</code><br/>
                • RAG LLM Backend: <code>{status.get('llm_backend')}</code><br/>
                • Avg Q&A Latency: <code>{qa_stats.get('avg_qa_latency_ms', 0.0)} ms</code>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# 2. UPLOAD RESEARCH PAPERS
# ---------------------------------------------------------------------------
def render_upload_papers(client: ScholarMindClient) -> None:
    st.header("Upload & Manage Research Papers")
    st.caption(
        "Extracts text page-by-page via PyMuPDF, preserves page provenance, performs scientific sentence-aligned "
        "chunking, computes `all-MiniLM-L6-v2` embeddings (cached), and persists to SQLite + FAISS."
    )

    uploaded_files = st.file_uploader(
        "Select one or more scientific PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process & Index Uploaded PDFs", type="primary"):
        for up in uploaded_files:
            try:
                with st.spinner(f"Extracting, chunking & indexing `{up.name}`..."):
                    res = client.upload_pdf(up.getvalue(), up.name)
                if res.get("already_indexed"):
                    st.info(
                        f"Skipped recomputation — **{res['document_name']}** is already indexed "
                        f"(`{res['document_id']}`, {res['chunks_indexed']} chunks)."
                    )
                else:
                    st.success(
                        f"Indexed **{res['document_name']}** — {res['page_count']} pages, "
                        f"{res['word_count']:,} words, {res['chunks_indexed']} FAISS chunks."
                    )
            except Exception as exc:
                st.error(f"Could not process `{up.name}`: {exc}")

    st.divider()
    docs = client.list_documents()
    st.subheader(f"Indexed Research Library ({len(docs)} documents)")
    if not docs:
        st.info("Upload a scientific PDF above to populate the library.")
        return

    for d in docs:
        with st.expander(f"📄 {d['document_name']} — {d['title']} ({d['page_count']} pages, {d['chunk_count']} chunks)"):
            c1, c2 = st.columns([4, 1])
            c1.markdown(
                f"- **Document ID:** `{d['document_id']}`\n"
                f"- **Authors:** {d.get('authors') or 'Not specified in PDF metadata'}\n"
                f"- **Words:** {d['word_count']:,} · **Chunks Indexed:** {d['chunk_count']}"
            )
            if c2.button("Delete Paper", key=f"delete_doc_{d['document_id']}", type="secondary"):
                res = client.delete_document(d["document_id"])
                st.warning(
                    f"Deleted `{d['document_name']}` and removed {res.get('vectors_removed', 0)} FAISS vectors."
                )
                st.rerun()

            if st.button("Inspect Extracted Pages & Sections", key=f"inspect_{d['document_id']}"):
                detail = client.get_paper_detail(d["document_id"])
                st.markdown("**Detected Sections:** " + ", ".join(f"`{s['section_title']}` (p.{s['page_number']})" for s in detail.get("sections", [])[:15]))
                for pg in detail.get("pages", [])[:5]:
                    st.text_area(
                        f"Page {pg['page_number']} ({pg['word_count']} words)",
                        value=pg["text"],
                        height=140,
                        disabled=True,
                        key=f"pg_{d['document_id']}_{pg['page_number']}",
                    )


# ---------------------------------------------------------------------------
# 3. PAPER SUMMARIZATION
# ---------------------------------------------------------------------------
def render_summarization(client: ScholarMindClient) -> None:
    st.header("T5 Scientific Paper Summarization")
    st.caption(
        "Integrates our Experiment D fine-tuned `google/flan-t5-small` checkpoint (`exp_d_t5/best_model.pt`) "
        "with chunk-based hierarchical Map-Reduce for long research papers."
    )

    docs = client.list_documents()
    selected = _doc_selector(docs, "Select paper to summarize", key="summ_doc_select")
    if not selected:
        return

    c1, c2 = st.columns(2)
    strategy = c1.selectbox(
        "Summarization Strategy",
        options=["auto", "hierarchical", "single_pass"],
        index=0,
        help="'hierarchical' summarizes representative document chunks and synthesizes them for long PDFs.",
    )
    max_chunks = c2.slider("Max Chunks (Hierarchical Map-Reduce)", min_value=2, max_value=8, value=4)

    if st.button("Generate Summary", type="primary"):
        with st.spinner("Running T5 summarization..."):
            res = client.summarize_paper(
                document_id=selected["document_id"],
                strategy=strategy,
                max_chunks=max_chunks,
            )
        st.session_state["last_summary"] = res

    if "last_summary" in st.session_state:
        res = st.session_state["last_summary"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Strategy", res["strategy"])
        m2.metric("Model", res["model_name"].split("/")[-1])
        m3.metric("Fine-Tuned Ckpt", "Yes (Exp D)" if res["used_finetuned_checkpoint"] else "Pretrained")
        m4.metric("Chunks Processed", res["chunks_processed"])

        st.markdown(f"### Summary: {res['paper_title']}")
        st.markdown(f"<div class='sm-card'>{res['summary']}</div>", unsafe_allow_html=True)
        st.download_button(
            "Download Summary (Markdown)",
            data=f"# Summary: {res['paper_title']}\n\n{res['summary']}\n",
            file_name=f"{res['paper_id']}_summary.md",
            mime="text/markdown",
        )


# ---------------------------------------------------------------------------
# 4. ASK RESEARCH ASSISTANT (MODULE 5 RAG)
# ---------------------------------------------------------------------------
def _render_rag_turn(turn: dict[str, Any], idx: int) -> None:
    with st.chat_message("user"):
        st.markdown(turn["question"])

    with st.chat_message("assistant"):
        status_key = turn.get("retrieval_status", "insufficient_evidence")
        badge_text, badge_color = STATUS_BADGES.get(status_key, ("Unknown", "gray"))
        passages = turn.get("retrieved_passages", [])
        sources = turn.get("sources", [])
        lat = turn.get("latencies_ms", {})

        c1, c2, c3 = st.columns([2, 1, 1])
        c1.markdown(f"**Retrieval Quality:** :{badge_color}[{badge_text}]")
        c2.metric("Top Cosine Similarity", f"{turn.get('max_similarity_score', 0.0):.3f}")
        c3.metric("Passages Retrieved", len(passages))
        st.caption(turn.get("evidence_note", ""))

        if turn.get("abstained"):
            st.warning(turn["answer"])
        else:
            st.markdown(turn["answer"])

        if sources:
            st.markdown("**Supporting Source Citations:**")
            for src in sources:
                pg = f"Page {src['page_number']}" if src.get("page_number") is not None else "Page unavailable"
                st.markdown(
                    f"- **[{src.get('rank', 1)}] `{src.get('document_name') or src.get('paper_title')}`** "
                    f"· **{pg}** (`{src.get('section_title', 'Body')}`) · "
                    f"Cosine Sim: `{src.get('similarity_score', 0.0):.3f}`\n"
                    f"  > {src.get('quote', '')}"
                )

        if passages:
            with st.expander(
                f"Retrieval Evidence Panel ({len(passages)} passages · "
                f"Retrieval {lat.get('retrieval_ms', 0):.0f} ms · LLM {lat.get('generation_ms', 0):.0f} ms)"
            ):
                for p in passages:
                    pg = f"Page {p['page_number']}" if p.get("page_number") else "Page unavailable"
                    st.markdown(
                        f"**Rank #{p.get('rank', 1)}** · `{p.get('document_name') or p.get('paper_title')}` · "
                        f"**{pg}** · Section: `{p.get('section_title')}` · "
                        f"Cosine Sim: `{p.get('dense_score', 0.0):.4f}` · BM25: `{p.get('lexical_score', 0.0):.2f}`"
                    )
                    st.text_area(
                        label=f"rag_ev_{idx}_{p['chunk_id']}",
                        value=p.get("text", ""),
                        height=105,
                        disabled=True,
                        label_visibility="collapsed",
                    )


def render_ask_assistant(client: ScholarMindClient) -> None:
    st.header("Ask Research Assistant (Grounded RAG)")
    docs = client.list_documents()
    if not docs:
        st.info("Upload at least one PDF in **2. Upload Research Papers** before asking questions.")
        return

    if "rag_history" not in st.session_state:
        st.session_state["rag_history"] = []
    if "pending_question" not in st.session_state:
        st.session_state["pending_question"] = ""

    # Document scope & retrieval controls
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1, 1, 1])
    scope = ctrl1.radio(
        "Document Scope",
        ["All uploaded papers", "Selected papers only"],
        horizontal=True,
    )
    top_k = ctrl2.slider("Top-K", 3, 8, 4)
    mode = ctrl3.selectbox("Mode", ["hybrid", "dense", "lexical"], index=0)
    sim_thresh = ctrl4.slider("Min Cosine Gate", 0.15, 0.70, 0.34, 0.02)

    selected_ids: list[str] | None = None
    if scope == "Selected papers only":
        doc_map = {f"{d['document_name']} ({d['page_count']} pp.)": d["document_id"] for d in docs}
        picked = st.multiselect("Select papers", list(doc_map.keys()), default=list(doc_map.keys())[:1])
        selected_ids = [doc_map[p] for p in picked]

    st.markdown("##### Quick Scientific Prompts")
    q_cols = st.columns(4)
    for i, q_text in enumerate(QUICK_RESEARCH_QUESTIONS):
        if q_cols[i % 4].button(q_text, key=f"qp_{i}", use_container_width=True):
            st.session_state["pending_question"] = q_text

    with st.form("ask_form", clear_on_submit=True):
        q_input = st.text_input(
            "Ask a technical or general research question:",
            value=st.session_state.get("pending_question", ""),
            placeholder="e.g., What datasets, architectures, and limitations did the authors report?",
        )
        col_sub, col_clr = st.columns([4, 1])
        submitted = col_sub.form_submit_button("Ask Research Assistant", type="primary", use_container_width=True)
        cleared = col_clr.form_submit_button("Clear History", use_container_width=True)

    if cleared:
        st.session_state["rag_history"] = []
        st.session_state["pending_question"] = ""
        st.rerun()

    if submitted and q_input.strip():
        st.session_state["pending_question"] = ""
        if scope == "Selected papers only" and not selected_ids:
            st.warning("Select at least one paper or switch scope to 'All uploaded papers'.")
        else:
            with st.spinner("Retrieving relevant passages from FAISS and generating grounded answer..."):
                res = client.ask_question(
                    question=q_input.strip(),
                    top_k=top_k,
                    mode=mode,
                    document_ids=selected_ids,
                    similarity_threshold=sim_thresh,
                )
            st.session_state["rag_history"].append(res)

    for idx, turn in enumerate(reversed(st.session_state["rag_history"])):
        _render_rag_turn(turn, idx)


# ---------------------------------------------------------------------------
# 5. SEMANTIC SEARCH
# ---------------------------------------------------------------------------
def render_semantic_search(client: ScholarMindClient) -> None:
    st.header("Hybrid Semantic Search (FAISS + BM25)")
    docs = client.list_documents()
    if not docs:
        st.info("Upload research papers first in **2. Upload Research Papers**.")
        return

    query = st.text_input(
        "Enter search query:",
        placeholder="e.g., transient global attention memory scaling ROUGE",
    )
    c1, c2, c3, c4 = st.columns([1.2, 1, 1.5, 1.5])
    mode = c1.selectbox("Search Mode", ["hybrid", "dense", "lexical"], index=0, key="search_mode")
    top_k = c2.slider("Top-K Results", 1, 15, 5, key="search_topk")
    doc_map = {d["document_name"]: d["document_id"] for d in docs}
    filter_docs = c3.multiselect("Filter Papers (optional)", list(doc_map.keys()))
    filter_secs = c4.multiselect("Filter Section Types (optional)", SECTION_TYPES)

    if st.button("Search Passages", type="primary") and query.strip():
        doc_ids = [doc_map[name] for name in filter_docs] if filter_docs else None
        sec_types = filter_secs if filter_secs else None
        with st.spinner("Searching FAISS + BM25 index..."):
            hits = client.semantic_search(
                query=query.strip(),
                top_k=top_k,
                mode=mode,
                document_ids=doc_ids,
                section_types=sec_types,
            )
        if not hits:
            st.warning("No matching passages found for this query and filter.")
            return
        st.success(f"Found {len(hits)} relevant passage(s).")
        for h in hits:
            st.markdown(
                f"""
                <div class="sm-card">
                    <strong>Rank #{h['rank']}</strong> · <code>{h.get('document_name') or h['paper_title']}</code> ·
                    <strong>Page {h['page_number']}</strong> · Section: <code>{h['section_title']}</code> (<em>{h['section_type']}</em>)<br/>
                    <small>RRF/Final Score: <code>{h['score']:.4f}</code> · Cosine Similarity: <code>{h['dense_score']:.4f}</code> · BM25: <code>{h['lexical_score']:.2f}</code></small>
                    <p style="margin-top:0.5rem;">{h['text']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# 6. RESEARCH GAP ANALYSIS
# ---------------------------------------------------------------------------
def render_gap_analysis(client: ScholarMindClient) -> None:
    st.header("Grounded Research & Gap Analysis")
    st.caption(
        "Extracts verifiable objectives, methodologies, datasets, architectures, findings, explicit limitations, "
        "and evidence-tied future directions directly from paper text — zero fabricated gaps."
    )
    docs = client.list_documents()
    selected = _doc_selector(docs, "Select paper to analyze", key="gap_doc_select")
    if not selected:
        return

    c1, c2 = st.columns([3, 1])
    run_btn = c1.button("Run Grounded Research Analysis", type="primary")
    refresh = c2.checkbox("Force refresh cache", value=False)

    if run_btn:
        with st.spinner("Analyzing paper structure, findings, and limitations..."):
            report = client.analyze_paper(selected["document_id"], refresh=refresh)
        st.session_state["last_analysis"] = report

    if "last_analysis" in st.session_state:
        rep = st.session_state["last_analysis"]
        st.subheader(rep["paper_title"])

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Detected Datasets & Benchmarks")
            if rep["datasets"]:
                st.markdown("".join(f"<span class='sm-pill'>{d}</span>" for d in rep["datasets"]), unsafe_allow_html=True)
            else:
                st.caption("None explicitly detected in extracted text.")
        with col_b:
            st.markdown("#### Detected Model Architectures")
            if rep["model_architectures"]:
                st.markdown("".join(f"<span class='sm-pill'>{m}</span>" for m in rep["model_architectures"]), unsafe_allow_html=True)
            else:
                st.caption("None explicitly detected in extracted text.")

        def _render_items(title: str, items: list[dict[str, Any]]) -> None:
            st.markdown(f"#### {title}")
            if not items:
                st.caption("No explicit statements detected in extracted text.")
                return
            for it in items:
                tag = "Author-Stated" if it["source_type"] == "author_stated" else "Derived from Stated Limitation"
                st.markdown(f"- **[{tag} · p. {it['page_number']}, {it['section_title']}]** {it['statement']}")

        _render_items("Objectives", rep.get("objectives", []))
        _render_items("Methodologies", rep.get("methodologies", []))
        _render_items("Key Findings & Reported Results", rep.get("key_findings", []))
        _render_items("Explicit Limitations", rep.get("explicit_limitations", []))
        _render_items("Potential Future Research Directions", rep.get("future_directions", []))

        if rep.get("extraction_notes"):
            st.info(" **Extraction Notes:** " + " | ".join(rep["extraction_notes"]))


# ---------------------------------------------------------------------------
# 7. PAPER COMPARISON
# ---------------------------------------------------------------------------
def render_paper_comparison(client: ScholarMindClient) -> None:
    st.header("Multi-Paper Comparison Matrix")
    docs = client.list_documents()
    if len(docs) < 2:
        st.info("Upload at least **2 research papers** in **2. Upload Research Papers** to compare them.")
        return

    doc_map = {f"{d['document_name']} — {d['title'][:50]}": d["document_id"] for d in docs}
    selected_labels = st.multiselect(
        "Select 2 to 10 papers to compare",
        options=list(doc_map.keys()),
        default=list(doc_map.keys())[:2],
    )

    if st.button("Compare Selected Papers", type="primary"):
        if len(selected_labels) < 2:
            st.warning("Select at least 2 papers.")
            return
        ids = [doc_map[lbl] for lbl in selected_labels]
        with st.spinner("Building grounded comparison matrix..."):
            cmp_res = client.compare_papers(ids)
        st.session_state["last_comparison"] = cmp_res

    if "last_comparison" in st.session_state:
        cmp_res = st.session_state["last_comparison"]
        st.markdown("#### Grounded Comparative Summary")
        st.markdown(f"<div class='sm-card'>{cmp_res['comparative_summary']}</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.markdown("**Shared Datasets:** " + (", ".join(f"`{x}`" for x in cmp_res["shared_datasets"]) or "None shared across all selected papers"))
        c2.markdown("**Shared Models:** " + (", ".join(f"`{x}`" for x in cmp_res["shared_models"]) or "None shared across all selected papers"))

        rows = []
        for entry in cmp_res["comparison_matrix"]:
            rows.append(
                {
                    "Paper Title": entry["paper_title"],
                    "Pages": entry["page_count"],
                    "Primary Objective": entry["primary_objective"][:180],
                    "Datasets": ", ".join(entry["datasets"]) or "—",
                    "Architectures": ", ".join(entry["model_architectures"]) or "—",
                    "Reported Results": " | ".join(entry["reported_results"][:2]) or "—",
                    "Explicit Limitations": " | ".join(entry["explicit_limitations"][:2]) or "—",
                }
            )
        df_cmp = pd.DataFrame(rows)
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)
        st.download_button(
            "Export Comparison Matrix (CSV)",
            data=df_cmp.to_csv(index=False),
            file_name="paper_comparison_matrix.csv",
            mime="text/csv",
        )

        with st.expander(f"Supporting Source Citations ({len(cmp_res.get('citations', []))})"):
            for cit in cmp_res.get("citations", []):
                st.markdown(
                    f"- **{cit['paper_title']}** (p. {cit['page_number']}, `{cit['section_title']}`): {cit['quote']}"
                )


# ---------------------------------------------------------------------------
# 8. DEEP LEARNING EXPERIMENTS
# ---------------------------------------------------------------------------
def render_dl_experiments(client: ScholarMindClient) -> None:
    st.header("Deep Learning & NLP Experiments (PyTorch, Seed = 42)")
    st.caption("All values below are loaded directly from saved empirical JSON artifacts — zero fabricated metrics.")

    res = client.get_experiment_results()
    plots = _plots_dir()

    # Classification Progression Table (Exp A, B, C)
    st.subheader("Experiments A, B & C: ArXiv 8-Class Scientific Domain Classification")
    cls_rows: list[dict[str, Any]] = []
    if "exp_a_rnn" in res:
        exp_a = res["exp_a_rnn"]
        maj = exp_a["baselines"]["majority_baseline"]
        tf = exp_a["baselines"]["tfidf_logreg_baseline"]
        cls_rows.append({"Model": "Baseline: Majority Class", "Val Loss": None, "Val Acc": maj["val_accuracy"], "Test Acc": maj["test_accuracy"], "Test Macro-F1": maj["test_macro_f1"], "Latency (ms)": None})
        cls_rows.append({"Model": "Baseline: TF-IDF + LogReg", "Val Loss": None, "Val Acc": tf["val_accuracy"], "Test Acc": tf["test_accuracy"], "Test Macro-F1": tf["test_macro_f1"], "Latency (ms)": None})
        for lbl, key in [("Exp A: Vanilla RNN (last h_T)", "vanilla_rnn_last"), ("Exp A: Vanilla RNN (mean pool)", "vanilla_rnn_mean")]:
            r = exp_a[key]
            cls_rows.append({"Model": lbl, "Val Loss": r["best_val"]["loss"], "Val Acc": r["best_val"]["accuracy"], "Test Acc": r["test"]["accuracy"], "Test Macro-F1": r["test"]["macro_f1"], "Latency (ms)": r["test"]["inference_ms_per_sample"]})
    if "exp_b_lstm_bilstm" in res:
        for lbl, key in [("Exp B: Unidirectional LSTM", "lstm"), ("Exp B: 2-Layer BiLSTM + Attn", "bilstm_attention")]:
            r = res["exp_b_lstm_bilstm"][key]
            cls_rows.append({"Model": lbl, "Val Loss": r["best_val"]["loss"], "Val Acc": r["best_val"]["accuracy"], "Test Acc": r["test"]["accuracy"], "Test Macro-F1": r["test"]["macro_f1"], "Latency (ms)": r["test"]["inference_ms_per_sample"]})
    if "exp_c_bert" in res:
        r = res["exp_c_bert"]["scibert_finetune"]
        cls_rows.append({"Model": "Exp C: SciBERT (top-2 layers + head)", "Val Loss": r["best_val"]["loss"], "Val Acc": r["best_val"]["accuracy"], "Test Acc": r["test"]["accuracy"], "Test Macro-F1": r["test"]["macro_f1"], "Latency (ms)": r["test"]["inference_ms_per_sample"]})

    if cls_rows:
        st.dataframe(pd.DataFrame(cls_rows), use_container_width=True, hide_index=True)

    p_cols = st.columns(3)
    for col, fname, cap in [
        (p_cols[0], "exp_a_rnn_curves.png", "Exp A: Vanilla RNN"),
        (p_cols[1], "exp_b_lstm_bilstm_curves.png", "Exp B: LSTM vs BiLSTM+Attn"),
        (p_cols[2], "exp_c_bert_curves.png", "Exp C: SciBERT Fine-Tuning"),
    ]:
        fpath = plots / fname
        if fpath.exists():
            col.image(str(fpath), caption=cap, use_container_width=True)

    if "exp_c_bert" in res:
        st.markdown("##### SciBERT Contextual Disambiguation Probes")
        st.dataframe(
            pd.DataFrame(res["exp_c_bert"]["contextual_embedding_probe"]["probes"]),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    # Experiment D: T5
    if "exp_d_t5" in res:
        st.subheader("Experiment D: T5 Transformer Summarization (`ccdv/arxiv-summarization`)")
        d = res["exp_d_t5"]
        t5_rows = [
            {"Method": "Baseline: Extractive Lead-3", **d["lead3_baseline_rouge"]},
            {"Method": f"Pretrained {d['model_name']} (Zero-Shot)", **d["pretrained_t5_rouge"]},
            {"Method": f"Pretrained {d['model_name']} (Hierarchical Map-Reduce)", "rouge1": d["hierarchical_map_reduce_rouge"]["rouge1"], "rouge2": d["hierarchical_map_reduce_rouge"]["rouge2"], "rougeL": d["hierarchical_map_reduce_rouge"]["rougeL"]},
            {"Method": f"PyTorch Fine-Tuned {d['model_name']} (Epoch {d['finetuning']['best_epoch']})", **d["finetuned_t5_rouge"]},
        ]
        c_left, c_right = st.columns([1.3, 1])
        c_left.dataframe(pd.DataFrame(t5_rows), use_container_width=True, hide_index=True)
        t5_plot = plots / "exp_d_t5_curves.png"
        if t5_plot.exists():
            c_right.image(str(t5_plot), caption="Exp D: T5 Fine-Tuning Loss", use_container_width=True)

    st.divider()
    # Experiment E: Embeddings
    if "exp_e_embeddings" in res:
        st.subheader("Experiment E: Embedding Comparison for Scientific Retrieval")
        e = res["exp_e_embeddings"]
        tab1, tab2 = st.tabs(["ArXiv Title → Abstract Retrieval", "SciQ Question → Support Retrieval"])
        tab1.dataframe(pd.DataFrame(e["arxiv_title_to_abstract"]), use_container_width=True, hide_index=True)
        tab2.dataframe(pd.DataFrame(e["sciq_question_to_support"]), use_container_width=True, hide_index=True)
        e_plot = plots / "exp_e_embeddings_comparison.png"
        if e_plot.exists():
            st.image(str(e_plot), caption="Exp E: TF-IDF vs SciBERT vs Sentence Transformer", use_container_width=True)


# ---------------------------------------------------------------------------
# 9. MODEL EVALUATION
# ---------------------------------------------------------------------------
def render_model_evaluation(client: ScholarMindClient) -> None:
    st.header("RAG System & Dataset Evaluation")
    res = client.get_experiment_results()
    plots = _plots_dir()

    if "module5_rag_eval" in res:
        rag = res["module5_rag_eval"]
        ret = rag["retrieval_metrics"]
        ans = rag["answer_quality_metrics"]
        perf = rag["performance_metrics_ms"]
        k = rag["top_k"]

        st.subheader("Module 5: End-to-End Scientific RAG Evaluation (Measured on CPU)")
        st.caption(
            f"Embeddings: `{rag['embedding_model']}` · LLM: `{rag['llm_backend']}` · "
            f"{rag['num_answerable_queries']} answerable + {rag['num_unanswerable_queries']} unanswerable test queries."
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(f"Recall@{k}", f"{ret[f'recall_at_{k}']:.3f}")
        m2.metric("MRR", f"{ret['mrr']:.4f}")
        m3.metric("Answer Token-F1", f"{ans['answer_token_f1']:.4f}")
        m4.metric("Citation Accuracy", f"{ans['citation_correctness']:.3f}")
        m5.metric("Unsupported Rate", f"{ans['unsupported_answer_rate']:.3f}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Retrieval & Answer Grounding Metrics")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Metric": f"Precision@{k}", "Value": ret[f"precision_at_{k}"]},
                        {"Metric": f"Recall@{k}", "Value": ret[f"recall_at_{k}"]},
                        {"Metric": "Mean Reciprocal Rank (MRR)", "Value": ret["mrr"]},
                        {"Metric": "Answer Token-F1 vs Reference", "Value": ans["answer_token_f1"]},
                        {"Metric": "Key Technical Term Coverage", "Value": ans["key_term_coverage"]},
                        {"Metric": "Evidence Relevance", "Value": ans["evidence_relevance"]},
                        {"Metric": "Exact Doc+Page Citation Accuracy", "Value": ans["citation_correctness"]},
                        {"Metric": "Unsupported Answer Rate (Lower=Better)", "Value": ans["unsupported_answer_rate"]},
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        with col2:
            st.markdown("##### CPU Pipeline Latencies (ms)")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Stage": "PDF Parsing & Page Chunking", "Mean (ms)": perf["avg_doc_processing_ms"]},
                        {"Stage": "Cold Embedding & FAISS Index", "Mean (ms)": perf["avg_cold_embedding_and_index_ms"]},
                        {"Stage": "Cached Re-Upload (Dedup Skip)", "Mean (ms)": perf["avg_cached_reupload_ms"]},
                        {"Stage": "Hybrid Retrieval (FAISS + BM25)", "Mean (ms)": perf["avg_retrieval_latency_ms"]},
                        {"Stage": "LLM Generation (flan-t5-base CPU)", "Mean (ms)": perf["avg_answer_generation_latency_ms"]},
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("##### Per-Query Grounding & Abstention Audit")
        st.dataframe(pd.DataFrame(rag["per_case_results"]), use_container_width=True, hide_index=True)

    st.divider()
    if "eda_summary" in res:
        st.subheader("Exploratory Data Analysis (4 Reference Datasets)")
        eda = res["eda_summary"]
        eda_rows = [
            {
                "Dataset": name,
                "Analyzed Rows": rep.get("quality", {}).get("row_count", 0),
                "Exact Duplicates": rep.get("quality", {}).get("exact_duplicate_rows", 0),
                "Splits": str(rep.get("split_sizes", "Single Catalog")),
            }
            for name, rep in eda.items()
        ]
        st.dataframe(pd.DataFrame(eda_rows), use_container_width=True, hide_index=True)

        ecols = st.columns(2)
        for idx, (fname, cap) in enumerate(
            [
                ("arxiv_eda.png", "ArXiv Metadata EDA"),
                ("squad_v2_eda.png", "SQuAD 2.0 EDA"),
                ("arxiv_summarization_eda.png", "ArXiv Summarization EDA"),
                ("sciq_eda.png", "SciQ EDA"),
            ]
        ):
            fpath = plots / fname
            if fpath.exists():
                ecols[idx % 2].image(str(fpath), caption=cap, use_container_width=True)


# ---------------------------------------------------------------------------
# 10. ABOUT PROJECT
# ---------------------------------------------------------------------------
def render_about_project() -> None:
    st.header("About ScholarMind AI")
    st.markdown(
        """
        ### Project Overview
        **ScholarMind AI** is an end-to-end **AI-Powered Research Paper Intelligence Assistant** engineered to demonstrate
        production NLP, PyTorch Deep Learning, Transformers, Embeddings, and Grounded Retrieval-Augmented Generation (RAG)
        on ordinary CPU laptops (with automatic GPU acceleration when available).

        ### Architecture Pipeline
        `PDF Upload` → `PyMuPDF Page-Preserving Extraction` → `Scientific NLP Cleaning & Section-Aware Chunking`
        → `Cached all-MiniLM-L6-v2 Embeddings` → `Persistent FAISS + BM25 Hybrid Retrieval`
        → `Similarity Guardrail & Grounded Open-Source LLM (Flan-T5 / Ollama)` → `Verified Page-Level Citations`

        ### Reference Datasets
        1. **ArXiv Metadata:** https://www.kaggle.com/datasets/Cornell-University/arxiv
        2. **SQuAD 2.0:** https://huggingface.co/datasets/rajpurkar/squad_v2
        3. **ArXiv Summarization:** https://huggingface.co/datasets/ccdv/arxiv-summarization
        4. **SciQ:** https://huggingface.co/datasets/allenai/sciq

        ### Honest Model Training vs. Pretrained Usage Summary
        - **Trained from Scratch in PyTorch:** Vanilla RNN (`exp_a_rnn`), Unidirectional LSTM (`exp_b_lstm`), 2-Layer BiLSTM + Additive Attention (`exp_b_bilstm_attn`) on ArXiv splits.
        - **Fine-Tuned in PyTorch:** `allenai/scibert_scivocab_uncased` top-2 layers + head (`exp_c_bert` on ArXiv), `google/flan-t5-small` (`exp_d_t5` on ArXiv Summarization).
        - **Pretrained Inference Models:** `sentence-transformers/all-MiniLM-L6-v2` & `BAAI/bge-small-en-v1.5` (dense retrieval), `google/flan-t5-base` / Ollama open-weights LLMs (grounded RAG answer generation).

        ### Key Engineering Principles
        - **Zero Hallucination Policy:** Explicitly abstains when cosine similarity & lexical evidence fall below threshold; cosine similarity is never mislabeled as factual probability.
        - **Zero Leakage Splits:** Pre-split deduplication, group-aware splits, train-only RNN vocabulary, and automated leakage audits.
        - **Full Reproducibility:** Fixed random seed (`42`), checksummed Parquet manifests, persisted SQLite + FAISS stores, and 28+ automated tests.
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="ScholarMind AI — Research Paper Intelligence Assistant",
        page_icon="📚",
        layout="wide",
    )
    _inject_styles()
    client = _get_client()

    with st.sidebar:
        st.title("📚 ScholarMind AI")
        st.caption("AI-Powered Research Paper Intelligence Assistant")
        st.divider()
        selected_section = st.radio("Navigation", SECTIONS, index=0)
        st.divider()
        st.caption(f"**Connection:** {client.connection_label}")

    if selected_section == SECTIONS[0]:
        render_dashboard(client)
    elif selected_section == SECTIONS[1]:
        render_upload_papers(client)
    elif selected_section == SECTIONS[2]:
        render_summarization(client)
    elif selected_section == SECTIONS[3]:
        render_ask_assistant(client)
    elif selected_section == SECTIONS[4]:
        render_semantic_search(client)
    elif selected_section == SECTIONS[5]:
        render_gap_analysis(client)
    elif selected_section == SECTIONS[6]:
        render_paper_comparison(client)
    elif selected_section == SECTIONS[7]:
        render_dl_experiments(client)
    elif selected_section == SECTIONS[8]:
        render_model_evaluation(client)
    else:
        render_about_project()


if __name__ == "__main__":
    main()
