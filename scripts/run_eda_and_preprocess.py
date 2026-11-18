"""CLI script running full EDA, NLP preprocessing, chunking, and leakage-free splitting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scholarmind.core.config import load_config
from scholarmind.core.logging import get_logger
from scholarmind.data.arxiv_loader import get_kaggle_manual_instructions, load_arxiv_jsonl
from scholarmind.data.eda import DatasetEDAAnalyzer
from scholarmind.data.pipeline import DataPreprocessingPipeline
from scholarmind.data.storage import load_parquet

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EDA, NLP preprocessing, chunking, and leakage-free splitting."
    )
    parser.add_argument(
        "--scale",
        choices=["mini", "small", "medium", "full"],
        default="mini",
        help="Dataset scale matching downloaded raw samples.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["all", "arxiv", "squad_v2", "arxiv_summ", "sciq"],
        default=["all"],
    )
    return parser.parse_args()


def _resolve_arxiv_jsonl(cfg, scale: str) -> tuple[Path, str] | None:
    if cfg.arxiv_manual_json_path.exists():
        return cfg.arxiv_manual_json_path, "kaggle:Cornell-University/arxiv"
    api_path = cfg.raw_dir / "arxiv" / f"arxiv-sample-api-{scale}.jsonl"
    if api_path.exists():
        return api_path, "arxiv_api_sample:compatible_kaggle_schema"
    return None


def main() -> None:
    args = parse_args()
    cfg = load_config()
    cfg.ensure_directories()
    scale_cfg = cfg.get_scale(args.scale)

    selected = set(args.datasets)
    if "all" in selected:
        selected = {"arxiv", "squad_v2", "arxiv_summ", "sciq"}

    eda = DatasetEDAAnalyzer(reports_dir=cfg.reports_dir)
    pipeline = DataPreprocessingPipeline(config=cfg)
    eda_reports: dict[str, dict[str, Any]] = {}

    # 1. ArXiv Metadata
    if "arxiv" in selected:
        resolved = _resolve_arxiv_jsonl(cfg, args.scale)
        if resolved is None:
            logger.warning("ArXiv JSONL not found. Skipping ArXiv.")
            print(get_kaggle_manual_instructions(cfg.arxiv_manual_json_path))
        else:
            jsonl_path, source = resolved
            arxiv_df = load_arxiv_jsonl(
                jsonl_path=jsonl_path,
                target_categories=cfg.arxiv_target_categories,
                max_records=scale_cfg.arxiv_max_records,
            )
            eda_reports["arxiv_metadata"] = eda.analyze_arxiv(arxiv_df)
            pipeline.process_arxiv_metadata(arxiv_df, scale=args.scale, source=source)

    # 2. SQuAD 2.0
    if "squad_v2" in selected:
        squad_raw_dir = cfg.raw_dir / "squad_v2"
        squad_splits = {
            "train": load_parquet(squad_raw_dir / f"train_{args.scale}.parquet"),
            "val": load_parquet(squad_raw_dir / f"val_{args.scale}.parquet"),
        }
        eda_reports["squad_v2"] = eda.analyze_squad_v2(squad_splits)
        pipeline.process_squad_v2(squad_splits, scale=args.scale)

    # 3. ArXiv Summarization + Scientific Document Chunking
    if "arxiv_summ" in selected:
        summ_raw_dir = cfg.raw_dir / "arxiv_summarization"
        summ_splits = {
            "train": load_parquet(summ_raw_dir / f"train_{args.scale}.parquet"),
            "val": load_parquet(summ_raw_dir / f"val_{args.scale}.parquet"),
            "test": load_parquet(summ_raw_dir / f"test_{args.scale}.parquet"),
        }
        eda_reports["arxiv_summarization"] = eda.analyze_arxiv_summarization(summ_splits)
        pipeline.process_arxiv_summarization(summ_splits, scale=args.scale)

    # 4. SciQ
    if "sciq" in selected:
        sciq_raw_dir = cfg.raw_dir / "sciq"
        sciq_splits = {
            "train": load_parquet(sciq_raw_dir / f"train_{args.scale}.parquet"),
            "val": load_parquet(sciq_raw_dir / f"val_{args.scale}.parquet"),
            "test": load_parquet(sciq_raw_dir / f"test_{args.scale}.parquet"),
        }
        eda_reports["sciq"] = eda.analyze_sciq(sciq_splits)
        pipeline.process_sciq(sciq_splits, scale=args.scale)

    if eda_reports:
        eda.save_reports(
            eda_reports,
            markdown_path=cfg.project_root / "docs" / "EDA_REPORT.md",
        )

    logger.info("EDA and NLP preprocessing pipeline finished successfully!")


if __name__ == "__main__":
    main()
