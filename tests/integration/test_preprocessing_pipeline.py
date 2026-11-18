"""End-to-end integration tests for dataset loading, EDA, NLP preprocessing, chunking, and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scholarmind.core.config import load_config
from scholarmind.data.arxiv_loader import load_arxiv_jsonl
from scholarmind.data.eda import DatasetEDAAnalyzer
from scholarmind.data.pipeline import DataPreprocessingPipeline
from scholarmind.data.storage import load_manifest, load_parquet
from scholarmind.nlp.tokenization import Vocabulary


def _make_temp_config(tmp_path: Path):
    cfg = load_config()
    cfg.data_dir = tmp_path / "data"
    cfg.raw_dir = tmp_path / "data" / "raw"
    cfg.processed_dir = tmp_path / "data" / "processed"
    cfg.reports_dir = tmp_path / "data" / "reports"
    cfg.artifacts_dir = tmp_path / "artifacts"
    cfg.arxiv_manual_json_path = cfg.raw_dir / "arxiv" / "arxiv-metadata-oai-snapshot.json"
    cfg.ensure_directories()
    return cfg


def test_full_pipeline_integration_across_all_four_datasets(tmp_path: Path) -> None:
    cfg = _make_temp_config(tmp_path)
    eda = DatasetEDAAnalyzer(reports_dir=cfg.reports_dir)
    pipeline = DataPreprocessingPipeline(config=cfg)

    # 1. ArXiv JSONL (manual Kaggle schema)
    arxiv_jsonl = cfg.arxiv_manual_json_path
    with arxiv_jsonl.open("w", encoding="utf-8") as fh:
        for i in range(18):
            cat = "cs.CL" if i < 9 else "cs.LG"
            rec = {
                "id": f"2501.{i:05d}",
                "title": f"Neural Scaling Study {i} for Scientific Retrieval",
                "abstract": (
                    f"In paper {i}, Vaswani et al. demonstrate that sparse self-attention improves "
                    f"long-context scientific retrieval without degrading accuracy across benchmarks. "
                    f"However, compute overhead remains non-trivial on edge devices in experiment {i}."
                ),
                "authors": "A. Researcher, B. Scientist",
                "categories": f"{cat} cs.AI",
                "versions": [{"version": "v1", "created": "Mon, 10 Jan 2025 12:00:00 GMT"}],
            }
            fh.write(json.dumps(rec) + "\n")

    arxiv_df = load_arxiv_jsonl(arxiv_jsonl, target_categories=["cs.CL", "cs.LG"], max_records=50)
    assert len(arxiv_df) == 18

    arxiv_eda = eda.analyze_arxiv(arxiv_df)
    arxiv_manifest = pipeline.process_arxiv_metadata(arxiv_df, scale="mini", source="test_kaggle")
    assert arxiv_manifest.leakage_audit.passed is True
    assert sum(arxiv_manifest.split_counts.values()) == 18
    vocab = Vocabulary.load(cfg.processed_dir / "arxiv_metadata" / "rnn_vocab.json")
    assert len(vocab) > 4

    # 2. SQuAD 2.0
    squad_train_rows = []
    for i in range(12):
        ctx = f"Article {i} explains that SciBERT uses a scientific vocabulary for NLP tasks."
        ans = "scientific vocabulary"
        is_imp = i % 3 == 0
        squad_train_rows.append(
            {
                "id": f"sq_tr_{i}",
                "title": f"Title_{i}",
                "context": ctx,
                "question": f"What vocabulary does SciBERT use in study {i}?",
                "answer_text": "" if is_imp else ans,
                "answer_start": -1 if is_imp else ctx.index(ans),
                "all_answers": [] if is_imp else [ans],
                "is_impossible": is_imp,
            }
        )
    squad_val_rows = [
        {
            "id": "sq_va_1",
            "title": "Val_Title_1",
            "context": "LongT5 introduces transient global attention for long documents.",
            "question": "What attention mechanism does LongT5 introduce?",
            "answer_text": "transient global attention",
            "answer_start": 18,
            "all_answers": ["transient global attention"],
            "is_impossible": False,
        },
        {
            "id": "sq_va_2",
            "title": "Val_Title_2",
            "context": "DeBERTa uses disentangled relative positional attention.",
            "question": "What learning rate was used?",
            "answer_text": "",
            "answer_start": -1,
            "all_answers": [],
            "is_impossible": True,
        },
    ]
    squad_raw = {"train": pd.DataFrame(squad_train_rows), "val": pd.DataFrame(squad_val_rows)}
    squad_eda = eda.analyze_squad_v2(squad_raw)
    squad_manifest = pipeline.process_squad_v2(squad_raw, scale="mini")
    assert squad_manifest.leakage_audit.passed is True
    squad_test_df = load_parquet(cfg.processed_dir / "squad_v2" / "test.parquet")
    ans_row = squad_test_df[~squad_test_df["is_impossible"]].iloc[0]
    assert ans_row["context"][ans_row["answer_start"] : ans_row["answer_start"] + len(ans_row["answer_text"])] == ans_row["answer_text"]

    # 3. ArXiv Summarization + Scientific Document Chunking
    def _make_summ_df(prefix: str, count: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "paper_id": f"{prefix}_{i}",
                    "article": (
                        f"1. Introduction\n"
                        f"Paper {prefix}_{i} studies long document summarization using @xmath0 and @xcite. "
                        + ("Scientific models process extended context windows efficiently. " * 12)
                        + "\n2. Methodology\n"
                        + ("We employ transient global attention across hierarchical sections. " * 12)
                        + "\n3. Limitations\n"
                        + ("Memory scales with sequence length and domain shift remains unexplored. " * 6)
                    ),
                    "abstract": (
                        f"We present method {prefix}_{i} for scientific paper summarization using sparse "
                        f"attention and verify strong ROUGE gains across arXiv benchmarks."
                    ),
                }
                for i in range(count)
            ]
        )

    summ_raw = {
        "train": _make_summ_df("tr", 4),
        "val": _make_summ_df("va", 2),
        "test": _make_summ_df("te", 2),
    }
    summ_eda = eda.analyze_arxiv_summarization(summ_raw)
    summ_manifest, chunk_manifest = pipeline.process_arxiv_summarization(summ_raw, scale="mini")
    assert summ_manifest.leakage_audit.passed is True
    assert chunk_manifest.leakage_audit.passed is True
    chunks_tr = load_parquet(cfg.processed_dir / "arxiv_scientific_chunks" / "train.parquet")
    assert len(chunks_tr) > 4
    assert "limitations" in set(chunks_tr["section_type"])

    # 4. SciQ
    def _make_sciq_df(prefix: str, count: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "id": f"sciq_{prefix}_{i}",
                    "question": f"Which organelle produces ATP in cell study {prefix}_{i}?",
                    "correct_answer": "mitochondria",
                    "distractor1": "ribosome",
                    "distractor2": "golgi",
                    "distractor3": "lysosome",
                    "support": f"In biology case {prefix}_{i}, mitochondria generate most cellular ATP.",
                    "has_support": True,
                }
                for i in range(count)
            ]
        )

    sciq_raw = {
        "train": _make_sciq_df("tr", 4),
        "val": _make_sciq_df("va", 2),
        "test": _make_sciq_df("te", 2),
    }
    sciq_eda = eda.analyze_sciq(sciq_raw)
    sciq_manifest = pipeline.process_sciq(sciq_raw, scale="mini")
    assert sciq_manifest.leakage_audit.passed is True

    # Save combined EDA reports & verify outputs exist
    json_path, md_path = eda.save_reports(
        {
            "arxiv_metadata": arxiv_eda,
            "squad_v2": squad_eda,
            "arxiv_summarization": summ_eda,
            "sciq": sciq_eda,
        }
    )
    assert json_path.exists()
    assert md_path.exists()
    assert load_manifest(cfg.processed_dir / "sciq" / "manifest.json").leakage_audit.passed is True
