"""CLI script for acquiring manageable (and scalable) dataset samples."""

from __future__ import annotations

import argparse

from scholarmind.core.config import load_config
from scholarmind.core.logging import get_logger
from scholarmind.data.arxiv_loader import (
    fetch_arxiv_api_sample,
    get_kaggle_manual_instructions,
)
from scholarmind.data.arxiv_summ_loader import load_arxiv_summarization
from scholarmind.data.sciq_loader import load_sciq
from scholarmind.data.squad_loader import load_squad_v2
from scholarmind.data.storage import save_parquet

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download manageable dataset samples (or scale up) for ScholarMind AI."
    )
    parser.add_argument(
        "--scale",
        choices=["mini", "small", "medium", "full"],
        default="mini",
        help="Dataset sample scale (default: mini to avoid massive automatic downloads).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["all", "arxiv", "squad_v2", "arxiv_summ", "sciq"],
        default=["all"],
        help="Datasets to acquire.",
    )
    parser.add_argument(
        "--arxiv-source",
        choices=["auto", "kaggle", "api"],
        default="auto",
        help=(
            "'kaggle': verify manual Kaggle JSONL only; "
            "'api': fetch lightweight sample from public ArXiv Atom API; "
            "'auto': use manual Kaggle JSONL if present, else fetch small API sample."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    cfg.ensure_directories()
    scale_cfg = cfg.get_scale(args.scale)

    selected = set(args.datasets)
    if "all" in selected:
        selected = {"arxiv", "squad_v2", "arxiv_summ", "sciq"}

    logger.info("Starting dataset acquisition (scale=%s, datasets=%s)", args.scale, sorted(selected))

    # 1. ArXiv Metadata (Manual Kaggle primary + lightweight API sample option)
    if "arxiv" in selected:
        kaggle_path = cfg.arxiv_manual_json_path
        api_sample_path = cfg.raw_dir / "arxiv" / f"arxiv-sample-api-{args.scale}.jsonl"

        if kaggle_path.exists():
            logger.info("Found manual Kaggle ArXiv snapshot at: %s", kaggle_path)
        elif args.arxiv_source == "kaggle":
            print(get_kaggle_manual_instructions(kaggle_path))
        else:
            print(get_kaggle_manual_instructions(kaggle_path))
            per_cat = max(10, min(60, (scale_cfg.arxiv_max_records or 400) // max(1, len(cfg.arxiv_target_categories))))
            logger.info("Fetching manageable public ArXiv API sample (%d/category) for immediate workflow...", per_cat)
            fetch_arxiv_api_sample(
                output_path=api_sample_path,
                categories=cfg.arxiv_target_categories,
                per_category=per_cat,
                delay_seconds=0.5,
            )

    # 2. SQuAD 2.0 (~16 MB compressed on HF; title-diverse balanced sampling)
    if "squad_v2" in selected:
        squad_splits = load_squad_v2(
            max_train=scale_cfg.squad_v2_train,
            max_val=scale_cfg.squad_v2_val,
            streaming=False,
            seed=cfg.seed,
        )
        for split_name, df in squad_splits.items():
            save_parquet(df, cfg.raw_dir / "squad_v2" / f"{split_name}_{args.scale}.parquet")

    # 3. ArXiv Summarization (always streaming by default to avoid 7.6 GB full download)
    if "arxiv_summ" in selected:
        summ_splits = load_arxiv_summarization(
            max_train=scale_cfg.arxiv_summ_train,
            max_val=scale_cfg.arxiv_summ_val,
            max_test=scale_cfg.arxiv_summ_test,
            streaming=True,
        )
        for split_name, df in summ_splits.items():
            save_parquet(df, cfg.raw_dir / "arxiv_summarization" / f"{split_name}_{args.scale}.parquet")

    # 4. SciQ
    if "sciq" in selected:
        sciq_splits = load_sciq(
            max_train=scale_cfg.sciq_train,
            max_val=scale_cfg.sciq_val,
            max_test=scale_cfg.sciq_test,
            streaming=True,
        )
        for split_name, df in sciq_splits.items():
            save_parquet(df, cfg.raw_dir / "sciq" / f"{split_name}_{args.scale}.parquet")

    logger.info("Dataset acquisition completed for scale='%s'.", args.scale)


if __name__ == "__main__":
    main()
