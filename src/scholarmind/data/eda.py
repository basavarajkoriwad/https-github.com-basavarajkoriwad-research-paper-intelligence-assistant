"""Exploratory Data Analysis (EDA) engine for all four scientific NLP datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Headless rendering for CI / Windows / server environments
import matplotlib.pyplot as plt

from scholarmind.core.logging import get_logger
from scholarmind.data.splits import compute_text_hash

logger = get_logger(__name__)


def _series_length_stats(series: pd.Series) -> dict[str, Any]:
    """Compute character, word, and estimated subword token length distributions."""
    as_str = series.fillna("").astype(str)
    char_lens = as_str.str.len().to_numpy(dtype=np.float64)
    word_lens = as_str.str.split().str.len().to_numpy(dtype=np.float64)
    est_tokens = np.round(word_lens * 1.30)

    def _summarize(arr: np.ndarray) -> dict[str, float]:
        if len(arr) == 0:
            return {}
        return {
            "min": float(np.min(arr)),
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "p25": round(float(np.percentile(arr, 25)), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p75": round(float(np.percentile(arr, 75)), 2),
            "p90": round(float(np.percentile(arr, 90)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "max": float(np.max(arr)),
        }

    return {
        "char_length": _summarize(char_lens),
        "word_length": _summarize(word_lens),
        "estimated_subword_tokens": _summarize(est_tokens),
        "pct_gt_512_tokens": round(float(np.mean(est_tokens > 512) * 100.0), 2) if len(est_tokens) else 0.0,
        "pct_gt_4096_tokens": round(float(np.mean(est_tokens > 4096) * 100.0), 2) if len(est_tokens) else 0.0,
    }


def _missing_and_quality_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze nulls, empty strings, whitespace-only values, and exact duplicate rows."""
    col_stats: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        s = df[col]
        null_count = int(s.isna().sum())
        empty_count = 0
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            non_null = s.dropna()
            str_mask = non_null.map(lambda x: isinstance(x, str))
            str_vals = non_null[str_mask]
            empty_count = int((str_vals.str.strip() == "").sum())
        col_stats[col] = {
            "dtype": str(s.dtype),
            "null_count": null_count,
            "empty_or_whitespace_count": empty_count,
            "missing_pct": round(((null_count + empty_count) / max(1, len(df))) * 100.0, 2),
        }

    # Hashable subset for duplicate row count (Parquet list columns reload as numpy.ndarray)
    hashable_cols = [
        c
        for c in df.columns
        if not df[c].dropna().map(lambda v: isinstance(v, (list, dict, np.ndarray))).any()
    ]
    exact_dups = int(df.duplicated(subset=hashable_cols).sum()) if hashable_cols else 0

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "exact_duplicate_rows": exact_dups,
        "columns": col_stats,
    }


def _safe_sample_records(df: pd.DataFrame, n: int = 2, max_chars: int = 260) -> list[dict[str, Any]]:
    """Return truncated, JSON-serializable sample records for inspection."""
    samples: list[dict[str, Any]] = []
    for _, row in df.head(n).iterrows():
        item: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, str) and len(v) > max_chars:
                item[k] = v[:max_chars] + "..."
            elif isinstance(v, (np.integer,)):
                item[k] = int(v)
            elif isinstance(v, (np.floating,)):
                item[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                item[k] = bool(v)
            elif isinstance(v, np.ndarray):
                item[k] = v.tolist()
            elif isinstance(v, (list, dict, str, int, float, bool)) or v is None:
                item[k] = v
            else:
                item[k] = str(v)
        samples.append(item)
    return samples


class DatasetEDAAnalyzer:
    """Computes rich statistical EDA profiles, plots, and Markdown reports for each dataset."""

    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = reports_dir
        self.plots_dir = reports_dir / "plots"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def analyze_arxiv(self, df: pd.DataFrame) -> dict[str, Any]:
        report: dict[str, Any] = {
            "dataset": "arxiv_metadata",
            "quality": _missing_and_quality_stats(df),
            "text_lengths": {
                "title": _series_length_stats(df["title"]) if "title" in df.columns else {},
                "abstract": _series_length_stats(df["abstract"]) if "abstract" in df.columns else {},
            },
            "distributions": {},
            "samples": _safe_sample_records(df, n=2),
        }

        if "primary_category" in df.columns:
            report["distributions"]["primary_category_counts"] = (
                df["primary_category"].value_counts().head(15).to_dict()
            )
        if "num_categories" in df.columns:
            report["distributions"]["avg_categories_per_paper"] = round(
                float(df["num_categories"].mean()), 2
            )
            report["distributions"]["multi_label_pct"] = round(
                float((df["num_categories"] > 1).mean() * 100.0), 2
            )
        if "year" in df.columns:
            valid_years = df["year"].dropna().astype(int)
            if not valid_years.empty:
                report["distributions"]["year_range"] = [int(valid_years.min()), int(valid_years.max())]
                report["distributions"]["top_years"] = valid_years.value_counts().head(10).to_dict()
        if "abstract" in df.columns:
            hashes = df["abstract"].map(compute_text_hash)
            report["quality"]["normalized_abstract_duplicates"] = int(
                hashes[hashes != ""].duplicated().sum()
            )

        self._plot_arxiv(df)
        return report

    def analyze_squad_v2(self, splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
        combined = pd.concat(
            [df.assign(_split=name) for name, df in splits.items() if not df.empty],
            ignore_index=True,
        )
        report: dict[str, Any] = {
            "dataset": "squad_v2",
            "split_sizes": {k: int(len(v)) for k, v in splits.items()},
            "quality": _missing_and_quality_stats(combined.drop(columns=["_split"], errors="ignore")),
            "text_lengths": {
                "context": _series_length_stats(combined["context"]),
                "question": _series_length_stats(combined["question"]),
                "answer_text_answerable_only": _series_length_stats(
                    combined.loc[~combined["is_impossible"], "answer_text"]
                ),
            },
            "distributions": {
                "unanswerable_pct_overall": round(float(combined["is_impossible"].mean() * 100.0), 2),
                "unanswerable_pct_by_split": {
                    k: round(float(v["is_impossible"].mean() * 100.0), 2)
                    for k, v in splits.items()
                    if not v.empty
                },
                "unique_titles": int(combined["title"].nunique()),
                "avg_questions_per_title": round(
                    float(len(combined) / max(1, combined["title"].nunique())), 2
                ),
            },
            "samples": _safe_sample_records(combined.drop(columns=["_split"], errors="ignore"), n=2),
        }
        self._plot_squad(combined)
        return report

    def analyze_arxiv_summarization(self, splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
        combined = pd.concat(
            [df.assign(_split=name) for name, df in splits.items() if not df.empty],
            ignore_index=True,
        )
        art_words = combined["article"].fillna("").str.split().str.len()
        abs_words = combined["abstract"].fillna("").str.split().str.len()
        comp_ratio = (art_words / abs_words.replace(0, np.nan)).dropna()

        xmath_pct = float(combined["article"].str.contains(r"@xmath\d+", regex=True, na=False).mean() * 100.0)
        xcite_pct = float(combined["article"].str.contains(r"@xcite", regex=True, na=False).mean() * 100.0)
        xref_pct = float(combined["article"].str.contains(r"@xref", regex=True, na=False).mean() * 100.0)

        report: dict[str, Any] = {
            "dataset": "arxiv_summarization",
            "split_sizes": {k: int(len(v)) for k, v in splits.items()},
            "quality": _missing_and_quality_stats(combined.drop(columns=["_split"], errors="ignore")),
            "text_lengths": {
                "article": _series_length_stats(combined["article"]),
                "abstract": _series_length_stats(combined["abstract"]),
            },
            "distributions": {
                "compression_ratio_mean": round(float(comp_ratio.mean()), 2) if len(comp_ratio) else 0.0,
                "compression_ratio_median": round(float(comp_ratio.median()), 2) if len(comp_ratio) else 0.0,
                "pct_articles_with_xmath": round(xmath_pct, 2),
                "pct_articles_with_xcite": round(xcite_pct, 2),
                "pct_articles_with_xref": round(xref_pct, 2),
            },
            "samples": _safe_sample_records(combined.drop(columns=["_split"], errors="ignore"), n=2),
        }
        self._plot_arxiv_summ(art_words, abs_words)
        return report

    def analyze_sciq(self, splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
        combined = pd.concat(
            [df.assign(_split=name) for name, df in splits.items() if not df.empty],
            ignore_index=True,
        )
        non_empty_support = combined.loc[combined["has_support"], "support"]
        report: dict[str, Any] = {
            "dataset": "sciq",
            "split_sizes": {k: int(len(v)) for k, v in splits.items()},
            "quality": _missing_and_quality_stats(combined.drop(columns=["_split"], errors="ignore")),
            "text_lengths": {
                "question": _series_length_stats(combined["question"]),
                "correct_answer": _series_length_stats(combined["correct_answer"]),
                "support_when_present": _series_length_stats(non_empty_support),
            },
            "distributions": {
                "has_support_pct": round(float(combined["has_support"].mean() * 100.0), 2),
                "missing_support_count": int((~combined["has_support"]).sum()),
                "unique_support_passages": int(non_empty_support.map(compute_text_hash).nunique()),
                "avg_correct_answer_words": round(
                    float(combined["correct_answer"].str.split().str.len().mean()), 2
                ),
                "avg_distractor1_words": round(
                    float(combined["distractor1"].str.split().str.len().mean()), 2
                ),
            },
            "samples": _safe_sample_records(combined.drop(columns=["_split"], errors="ignore"), n=2),
        }
        self._plot_sciq(combined)
        return report

    def _plot_arxiv(self, df: pd.DataFrame) -> None:
        if df.empty or "abstract" not in df.columns:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        words = df["abstract"].fillna("").str.split().str.len()
        axes[0].hist(words, bins=30, color="#2563eb", alpha=0.85, edgecolor="white")
        axes[0].set_title("ArXiv Abstract Length (Words)")
        axes[0].set_xlabel("Word Count")
        axes[0].set_ylabel("Paper Count")

        if "primary_category" in df.columns:
            counts = df["primary_category"].value_counts().head(10)
            counts.plot(kind="barh", ax=axes[1], color="#0f766e")
            axes[1].invert_yaxis()
            axes[1].set_title("Top Primary ArXiv Categories")
            axes[1].set_xlabel("Paper Count")

        plt.tight_layout()
        fig.savefig(self.plots_dir / "arxiv_eda.png", dpi=140)
        plt.close(fig)

    def _plot_squad(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ctx_words = df["context"].fillna("").str.split().str.len()
        axes[0].hist(ctx_words, bins=30, color="#7c3aed", alpha=0.85, edgecolor="white")
        axes[0].set_title("SQuAD 2.0 Context Length (Words)")
        axes[0].set_xlabel("Word Count")
        axes[0].set_ylabel("Count")

        ans_counts = df["is_impossible"].map({False: "Answerable", True: "Unanswerable"}).value_counts()
        ans_counts.plot(kind="bar", ax=axes[1], color=["#16a34a", "#dc2626"])
        axes[1].set_title("SQuAD 2.0 Answerability Distribution")
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis="x", rotation=0)

        plt.tight_layout()
        fig.savefig(self.plots_dir / "squad_v2_eda.png", dpi=140)
        plt.close(fig)

    def _plot_arxiv_summ(self, art_words: pd.Series, abs_words: pd.Series) -> None:
        if art_words.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        clipped_art = art_words.clip(upper=np.percentile(art_words, 98))
        axes[0].hist(clipped_art, bins=30, color="#ea580c", alpha=0.85, edgecolor="white")
        axes[0].set_title("ArXiv Summarization Article Length (Words, <=P98)")
        axes[0].set_xlabel("Word Count")
        axes[0].set_ylabel("Paper Count")

        clipped_abs = abs_words.clip(upper=np.percentile(abs_words, 98))
        axes[1].hist(clipped_abs, bins=30, color="#0284c7", alpha=0.85, edgecolor="white")
        axes[1].set_title("ArXiv Summarization Abstract Length (Words, <=P98)")
        axes[1].set_xlabel("Word Count")

        plt.tight_layout()
        fig.savefig(self.plots_dir / "arxiv_summarization_eda.png", dpi=140)
        plt.close(fig)

    def _plot_sciq(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        sup_words = df.loc[df["has_support"], "support"].str.split().str.len()
        if not sup_words.empty:
            axes[0].hist(sup_words.clip(upper=np.percentile(sup_words, 98)), bins=25, color="#0891b2", alpha=0.85, edgecolor="white")
        axes[0].set_title("SciQ Support Passage Length (Words)")
        axes[0].set_xlabel("Word Count")
        axes[0].set_ylabel("Count")

        sup_counts = df["has_support"].map({True: "Has Support", False: "No Support"}).value_counts()
        sup_counts.plot(kind="bar", ax=axes[1], color=["#2563eb", "#94a3b8"])
        axes[1].set_title("SciQ Support Passage Availability")
        axes[1].tick_params(axis="x", rotation=0)

        plt.tight_layout()
        fig.savefig(self.plots_dir / "sciq_eda.png", dpi=140)
        plt.close(fig)

    def save_reports(self, all_reports: dict[str, dict[str, Any]], markdown_path: Path | None = None) -> tuple[Path, Path]:
        """Persist combined EDA summary as JSON and readable Markdown."""
        json_path = self.reports_dir / "eda_summary.json"
        json_path.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")

        md_path = markdown_path or (self.reports_dir / "EDA_REPORT.md")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_lines = [
            "# Exploratory Data Analysis (EDA) Report",
            "",
            "Generated automatically by `scholarmind.data.eda.DatasetEDAAnalyzer`.",
            "",
        ]

        for name, rep in all_reports.items():
            md_lines.append(f"## Dataset: `{name}`")
            md_lines.append("")
            quality = rep.get("quality", {})
            md_lines.append(f"- **Total Analyzed Rows:** {quality.get('row_count', 0)}")
            md_lines.append(f"- **Exact Duplicate Rows:** {quality.get('exact_duplicate_rows', 0)}")
            if "split_sizes" in rep:
                md_lines.append(f"- **Split Sizes:** `{rep['split_sizes']}`")
            md_lines.append("")

            md_lines.append("### Text Length Statistics (Word Count & Estimated Subword Tokens)")
            md_lines.append("")
            md_lines.append("| Field | Mean Words | Median (P50) | P95 Words | Max Words | >512 Tok (%) | >4096 Tok (%) |")
            md_lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for field, stats in rep.get("text_lengths", {}).items():
                wl = stats.get("word_length", {})
                if not wl:
                    continue
                md_lines.append(
                    f"| `{field}` | {wl.get('mean', 0)} | {wl.get('p50', 0)} | {wl.get('p95', 0)} | "
                    f"{wl.get('max', 0)} | {stats.get('pct_gt_512_tokens', 0)}% | {stats.get('pct_gt_4096_tokens', 0)}% |"
                )
            md_lines.append("")

            md_lines.append("### Missing & Empty Value Summary")
            md_lines.append("")
            md_lines.append("| Column | Dtype | Nulls | Empty/Whitespace | Missing % |")
            md_lines.append("| :--- | :--- | ---: | ---: | ---: |")
            for col, cstats in quality.get("columns", {}).items():
                md_lines.append(
                    f"| `{col}` | `{cstats['dtype']}` | {cstats['null_count']} | "
                    f"{cstats['empty_or_whitespace_count']} | {cstats['missing_pct']}% |"
                )
            md_lines.append("")

            dists = rep.get("distributions", {})
            if dists:
                md_lines.append("### Key Distributions & Domain Insights")
                md_lines.append("")
                for dk, dv in dists.items():
                    md_lines.append(f"- **`{dk}`:** `{dv}`")
                md_lines.append("")

            samples = rep.get("samples", [])
            if samples:
                md_lines.append("### Sample Record Preview")
                md_lines.append("```json")
                md_lines.append(json.dumps(samples[0], indent=2, ensure_ascii=False))
                md_lines.append("```")
                md_lines.append("")

        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        logger.info("Saved EDA reports -> %s and %s", json_path, md_path)
        return json_path, md_path
