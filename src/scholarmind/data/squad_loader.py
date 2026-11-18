"""Hugging Face loader for SQuAD 2.0 (rajpurkar/squad_v2) with title-diverse balanced sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from datasets import load_dataset

from scholarmind.core.logging import get_logger

logger = get_logger(__name__)

HF_SQUAD_V2_ID = "rajpurkar/squad_v2"


def _flatten_squad_record(record: dict) -> dict:
    answers = record.get("answers") or {}
    texts = answers.get("text") or []
    starts = answers.get("answer_start") or []
    is_impossible = len(texts) == 0

    return {
        "id": str(record.get("id", "")),
        "title": str(record.get("title", "")),
        "context": str(record.get("context", "")),
        "question": str(record.get("question", "")),
        "answer_text": str(texts[0]) if texts else "",
        "answer_start": int(starts[0]) if starts else -1,
        "all_answers": [str(t) for t in texts],
        "is_impossible": bool(is_impossible),
    }


def _sample_title_diverse(
    df: pd.DataFrame,
    limit: int | None,
    seed: int,
    max_per_title: int = 25,
) -> pd.DataFrame:
    """Sample records across many distinct Wikipedia titles while preserving answerability balance.

    SQuAD 2.0 orders records sequentially by article, with many unanswerable questions clustered
    in later articles. Naive prefix or small-buffer streaming yields 0% unanswerable examples
    and only 1-2 titles. Grouping by title and capping per-title samples guarantees realistic
    answerable/unanswerable ratios and enough independent groups for leakage-free splitting.
    """
    if limit is None or len(df) <= limit:
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    titles = rng.permutation(df["title"].unique())
    selected_chunks: list[pd.DataFrame] = []
    collected = 0

    for title in titles:
        sub = df[df["title"] == title]
        take = min(len(sub), max_per_title, limit - collected)
        if take <= 0:
            break
        selected_chunks.append(sub.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
        collected += take
        if collected >= limit:
            break

    return pd.concat(selected_chunks, ignore_index=True)


def load_squad_v2(
    max_train: int | None = 3000,
    max_val: int | None = 800,
    streaming: bool = False,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Load SQuAD 2.0 (~16 MB compressed Parquet on HF) with title-diverse balanced sampling."""
    splits: dict[str, pd.DataFrame] = {}

    for split_name, limit in [("train", max_train), ("validation", max_val)]:
        logger.info(
            "Loading %s [%s] (limit=%s, streaming=%s)...",
            HF_SQUAD_V2_ID,
            split_name,
            limit,
            streaming,
        )
        ds = load_dataset(HF_SQUAD_V2_ID, split=split_name, streaming=streaming)
        if streaming:
            # Large shuffle buffer when streaming is explicitly requested
            ds = ds.shuffle(seed=seed, buffer_size=25000)
            rows = []
            for idx, rec in enumerate(ds):
                rows.append(_flatten_squad_record(rec))
                if limit is not None and idx + 1 >= limit:
                    break
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame(_flatten_squad_record(rec) for rec in ds)
            df = _sample_title_diverse(df, limit=limit, seed=seed)

        out_key = "val" if split_name == "validation" else split_name
        splits[out_key] = df
        unans_pct = float(df["is_impossible"].mean() * 100.0) if len(df) else 0.0
        logger.info(
            "Loaded SQuAD 2.0 '%s': %d rows | %d unique titles | %.1f%% unanswerable.",
            out_key,
            len(df),
            df["title"].nunique() if len(df) else 0,
            unans_pct,
        )

    return splits
