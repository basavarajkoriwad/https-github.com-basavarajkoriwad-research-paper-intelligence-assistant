"""Export real benchmark plots & build a unified empirical summary figure for docs/screenshots/."""

from __future__ import annotations

import json
import shutil

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scholarmind.core.config import load_config


def main() -> None:
    cfg = load_config()
    plots_dir = cfg.reports_dir / "plots"
    shots_dir = cfg.project_root / "docs" / "screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)

    # Copy real experiment & EDA plots into tracked docs/screenshots/
    for fname in [
        "exp_a_rnn_curves.png",
        "exp_b_lstm_bilstm_curves.png",
        "exp_c_bert_curves.png",
        "exp_d_t5_curves.png",
        "exp_e_embeddings_comparison.png",
        "arxiv_eda.png",
        "arxiv_summarization_eda.png",
    ]:
        src = plots_dir / fname
        if src.exists():
            shutil.copy2(src, shots_dir / fname)

    # Build a crisp 3-panel real empirical overview figure strictly from saved JSON results
    exp_dir = cfg.artifacts_dir / "experiments"
    exp_a = json.loads((exp_dir / "exp_a_rnn_results.json").read_text(encoding="utf-8"))
    exp_b = json.loads((exp_dir / "exp_b_lstm_bilstm_results.json").read_text(encoding="utf-8"))
    exp_c = json.loads((exp_dir / "exp_c_bert_results.json").read_text(encoding="utf-8"))
    exp_d = json.loads((exp_dir / "exp_d_t5_results.json").read_text(encoding="utf-8"))
    rag = json.loads((exp_dir / "module5_rag_eval.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel 1: Classification Macro-F1 Progression (Exp A-C)
    cls_labels = ["Majority", "RNN (h_T)", "LSTM", "TF-IDF+LR", "SciBERT"]
    cls_f1 = [
        exp_a["baselines"]["majority_baseline"]["test_macro_f1"],
        exp_a["vanilla_rnn_last"]["test"]["macro_f1"],
        exp_b["lstm"]["test"]["macro_f1"],
        exp_a["baselines"]["tfidf_logreg_baseline"]["test_macro_f1"],
        exp_c["scibert_finetune"]["test"]["macro_f1"],
    ]
    colors1 = ["#94a3b8", "#f97316", "#eab308", "#0ea5e9", "#16a34a"]
    bars1 = axes[0].bar(cls_labels, cls_f1, color=colors1, edgecolor="white")
    axes[0].set_title("Exp A–C: ArXiv Test Macro-F1")
    axes[0].set_ylim(0, 0.72)
    axes[0].set_ylabel("Macro-F1")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=18)
    for b, v in zip(bars1, cls_f1):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)

    # Panel 2: T5 Summarization ROUGE Progression (Exp D)
    t5_labels = ["Pretrained\nFlan-T5", "Fine-Tuned\nFlan-T5 (Ep3)", "Lead-3\nBaseline"]
    r1 = [
        exp_d["pretrained_t5_rouge"]["rouge1"],
        exp_d["finetuned_t5_rouge"]["rouge1"],
        exp_d["lead3_baseline_rouge"]["rouge1"],
    ]
    rl = [
        exp_d["pretrained_t5_rouge"]["rougeL"],
        exp_d["finetuned_t5_rouge"]["rougeL"],
        exp_d["lead3_baseline_rouge"]["rougeL"],
    ]
    x2 = np.arange(len(t5_labels))
    w = 0.34
    b_r1 = axes[1].bar(x2 - w / 2, r1, width=w, label="ROUGE-1", color="#2563eb")
    b_rl = axes[1].bar(x2 + w / 2, rl, width=w, label="ROUGE-L", color="#7c3aed")
    axes[1].set_title("Exp D: ArXiv Summarization ROUGE")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(t5_labels)
    axes[1].set_ylim(0, 0.36)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    for b, v in list(zip(b_r1, r1)) + list(zip(b_rl, rl)):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)

    # Panel 3: Module 5 Scientific RAG Evaluation
    rag_labels = ["Recall@4", "MRR", "Token-F1", "Evidence\nRel.", "Citation\nAcc.", "Abstention\nSafety"]
    rag_vals = [
        rag["retrieval_metrics"]["recall_at_4"],
        rag["retrieval_metrics"]["mrr"],
        rag["answer_quality_metrics"]["answer_token_f1"],
        rag["answer_quality_metrics"]["evidence_relevance"],
        rag["answer_quality_metrics"]["citation_correctness"],
        1.0 - rag["answer_quality_metrics"]["unsupported_answer_rate"],
    ]
    bars3 = axes[2].bar(rag_labels, rag_vals, color="#0f766e", edgecolor="white")
    axes[2].set_title("Module 5: Grounded RAG Evaluation")
    axes[2].set_ylim(0, 1.15)
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].tick_params(axis="x", rotation=18)
    for b, v in zip(bars3, rag_vals):
        axes[2].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5)

    plt.suptitle("ScholarMind AI — Measured Empirical Results Summary (CPU, Seed=42)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = shots_dir / "benchmarks_overview.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Exported visuals into {shots_dir}")


if __name__ == "__main__":
    main()
