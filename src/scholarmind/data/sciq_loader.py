"""Hugging Face loader for SciQ (allenai/sciq)."""

from __future__ import annotations

from itertools import islice

import pandas as pd
from datasets import load_dataset

from scholarmind.core.logging import get_logger

logger = get_logger(__name__)

HF_SCIQ_ID = "allenai/sciq"


def load_sciq(
    max_train: int | None = 2000,
    max_val: int | None = 500,
    max_test: int | None = 500,
    streaming: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load SciQ train/validation/test splits (~10 MB total, supports streaming caps)."""
    splits: dict[str, pd.DataFrame] = {}

    split_spec = [
        ("train", "train", max_train),
        ("validation", "val", max_val),
        ("test", "test", max_test),
    ]

    for hf_split, out_key, limit in split_spec:
        logger.info(
            "Loading %s [%s] (limit=%s, streaming=%s)...",
            HF_SCIQ_ID,
            hf_split,
            limit,
            streaming,
        )
        ds = load_dataset(HF_SCIQ_ID, split=hf_split, streaming=streaming)
        iterator = islice(ds, limit) if (streaming and limit is not None) else (
            ds.select(range(min(limit, len(ds)))) if limit is not None else ds
        )

        rows = []
        for idx, rec in enumerate(iterator):
            support = str(rec.get("support") or "").strip()
            rows.append(
                {
                    "id": f"sciq_{out_key}_{idx:05d}",
                    "question": str(rec.get("question") or ""),
                    "correct_answer": str(rec.get("correct_answer") or ""),
                    "distractor1": str(rec.get("distractor1") or ""),
                    "distractor2": str(rec.get("distractor2") or ""),
                    "distractor3": str(rec.get("distractor3") or ""),
                    "support": support,
                    "has_support": bool(support),
                }
            )
        splits[out_key] = pd.DataFrame(rows)
        logger.info("Loaded SciQ '%s': %d rows.", out_key, len(splits[out_key]))

    return splits
