"""Streaming Hugging Face loader for ArXiv Summarization (ccdv/arxiv-summarization)."""

from __future__ import annotations

from itertools import islice

import pandas as pd
from datasets import load_dataset

from scholarmind.core.logging import get_logger

logger = get_logger(__name__)

HF_ARXIV_SUMM_ID = "ccdv/arxiv-summarization"


def load_arxiv_summarization(
    max_train: int | None = 500,
    max_val: int | None = 100,
    max_test: int | None = 100,
    config_name: str = "section",
    streaming: bool = True,
) -> dict[str, pd.DataFrame]:
    """Stream manageable slices of `ccdv/arxiv-summarization` without downloading 7.6 GB.

    Defaults to `streaming=True` so small/mini samples pull only a few megabytes while
    scaling cleanly to the full 203K corpus when `max_*=None`.
    """
    splits: dict[str, pd.DataFrame] = {}

    split_spec = [
        ("train", "train", max_train),
        ("validation", "val", max_val),
        ("test", "test", max_test),
    ]

    for hf_split, out_key, limit in split_spec:
        logger.info(
            "Loading %s (%s) [%s] (limit=%s, streaming=%s)...",
            HF_ARXIV_SUMM_ID,
            config_name,
            hf_split,
            limit,
            streaming,
        )
        ds = load_dataset(HF_ARXIV_SUMM_ID, config_name, split=hf_split, streaming=streaming)
        iterator = islice(ds, limit) if (streaming and limit is not None) else (
            ds.select(range(min(limit, len(ds)))) if limit is not None else ds
        )

        rows = []
        for idx, rec in enumerate(iterator):
            rows.append(
                {
                    "paper_id": str(rec.get("id") or f"arxiv_summ_{out_key}_{idx}"),
                    "article": str(rec.get("article") or ""),
                    "abstract": str(rec.get("abstract") or ""),
                }
            )
        splits[out_key] = pd.DataFrame(rows)
        logger.info("Loaded ArXiv Summarization '%s': %d rows.", out_key, len(splits[out_key]))

    return splits
