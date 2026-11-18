"""Integration tests for Dashboard telemetry, system/status, experiments/results, and UI client."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
import pytest

from scholarmind.api.dependencies import ServiceContainer
from scholarmind.api.routes.system import get_experiment_results, system_status
from scholarmind.core.config import load_config
from scholarmind.ui import app as ui_app
from scholarmind.ui.api_client import ScholarMindClient


def _sample_pdf() -> bytes:
    doc = fitz.open()
    doc.set_metadata({"title": "TelemetryCheck: Scientific RAG", "author": "Test Author"})
    p = doc.new_page()
    p.insert_textbox(
        fitz.Rect(50, 50, 545, 780),
        "Abstract\nWe evaluate TelemetryCheck on ArXiv and SciQ using FAISS and all-MiniLM-L6-v2. "
        "Limitations: this prototype has not been evaluated on image-only scanned PDFs.",
        fontsize=11,
    )
    raw = doc.tobytes()
    doc.close()
    return raw


@pytest.mark.slow_model
def test_system_status_qa_logging_and_experiment_results_sanitization(tmp_path: Path) -> None:
    cfg = load_config()
    container = ServiceContainer(config=cfg, store_dir=tmp_path / "dash_store")

    # Verify ScholarMindClient in local engine mode works seamlessly across every workflow
    client = ScholarMindClient(prefer_api=False)
    client._container = container

    st0 = client.get_system_status()
    assert st0["indexed_papers"] == 0
    assert st0["qa_stats"]["questions_asked"] == 0
    assert len(st0["models_inventory"]) >= 5
    # Verify actual trained checkpoints from Stage 3 are detected on disk
    by_comp = {m["component"]: m for m in st0["models_inventory"]}
    assert by_comp["Experiment A: Vanilla RNN"]["available"] is True
    assert by_comp["Experiment C: SciBERT Classifier"]["available"] is True
    assert by_comp["Experiment D: T5 Summarizer"]["available"] is True

    # Upload -> Ask -> Verify Dashboard counters increment & persist in SQLite!
    up = client.upload_pdf(_sample_pdf(), "TelemetryCheck.pdf")
    assert up["document_name"] == "TelemetryCheck.pdf"

    ans = client.ask_question("Which datasets does TelemetryCheck evaluate on?", top_k=3)
    assert ans["abstained"] is False

    st1 = system_status(container)
    assert st1["indexed_papers"] == 1
    assert st1["indexed_vectors"] >= 1
    assert st1["qa_stats"]["questions_asked"] == 1
    assert len(st1["qa_stats"]["recent_questions"]) == 1

    # Verify experiment results endpoint returns real JSONs with NO local absolute file paths leaked
    exps = get_experiment_results(container)
    assert "exp_a_rnn" in exps
    assert "exp_d_t5" in exps
    assert "module5_rag_eval" in exps
    serialized = str(exps)
    assert "C:\\Users\\" not in serialized


def test_streamlit_dashboard_sections_defined() -> None:
    assert len(ui_app.SECTIONS) == 10
    expected_keywords = [
        "Dashboard",
        "Upload Research Papers",
        "Paper Summarization",
        "Ask Research Assistant",
        "Semantic Search",
        "Research Gap Analysis",
        "Paper Comparison",
        "Deep Learning Experiments",
        "Model Evaluation",
        "About Project",
    ]
    for section_label, kw in zip(ui_app.SECTIONS, expected_keywords):
        assert kw in section_label
