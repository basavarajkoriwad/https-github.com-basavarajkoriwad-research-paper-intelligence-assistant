"""Unit tests for deduplication, leakage-free splitting, leakage audits, and Parquet persistence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scholarmind.data.splits import (
    compute_text_hash,
    deduplicate_dataframe,
    group_aware_split,
    remove_cross_split_leakage,
    verify_no_leakage,
)
from scholarmind.data.storage import load_manifest, load_parquet, save_dataset_splits


def test_deduplicate_and_group_aware_split_prevents_leakage() -> None:
    rows = []
    # 12 distinct articles across 2 categories, 3 records per article + deliberate duplicates
    for art_idx in range(12):
        cat = "cs.CL" if art_idx % 2 == 0 else "cs.LG"
        for q_idx in range(3):
            rows.append(
                {
                    "id": f"rec_{art_idx}_{q_idx}",
                    "group": f"article_{art_idx}",
                    "category": cat,
                    "text": f"Unique scientific statement {art_idx} question {q_idx} about transformers.",
                }
            )
    # Add exact text duplicate with different ID
    rows.append(
        {
            "id": "rec_dup",
            "group": "article_0",
            "category": "cs.CL",
            "text": "Unique scientific statement 0 question 0 about transformers.",
        }
    )

    df = pd.DataFrame(rows)
    deduped, removed = deduplicate_dataframe(df, text_cols=["text"], id_col="id")
    assert removed == 1

    deduped["text_hash"] = deduped["text"].map(compute_text_hash)
    splits = group_aware_split(
        deduped,
        group_col="group",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        stratify_col="category",
    )

    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0

    audit = verify_no_leakage(
        dataset_name="synthetic_test",
        splits=splits,
        id_col="id",
        group_col="group",
        text_hash_col="text_hash",
        strict=True,
    )
    assert audit.passed is True


def test_verify_no_leakage_detects_and_removes_intentional_leak() -> None:
    train_df = pd.DataFrame(
        [
            {"id": "1", "group": "g1", "text_hash": "h1"},
            {"id": "2", "group": "g2", "text_hash": "h2"},  # Leaks group g2 into test!
        ]
    )
    test_df = pd.DataFrame(
        [
            {"id": "3", "group": "g2", "text_hash": "h3"},
        ]
    )

    with pytest.raises(AssertionError, match="Data leakage detected"):
        verify_no_leakage(
            dataset_name="leaky_ds",
            splits={"train": train_df, "test": test_df},
            id_col="id",
            group_col="group",
            text_hash_col="text_hash",
            strict=True,
        )

    cleaned, removed = remove_cross_split_leakage(
        {"train": train_df, "test": test_df},
        group_col="group",
        text_hash_col="text_hash",
        priority_order=("test", "train"),
    )
    assert removed == 1
    assert len(cleaned["test"]) == 1  # Higher-priority test set remains untouched
    assert len(cleaned["train"]) == 1

    audit = verify_no_leakage(
        dataset_name="fixed_ds",
        splits=cleaned,
        id_col="id",
        group_col="group",
        text_hash_col="text_hash",
        strict=True,
    )
    assert audit.passed is True


def test_parquet_and_manifest_roundtrip(tmp_path: Path) -> None:
    splits = {
        "train": pd.DataFrame({"id": ["a", "b"], "val": [1, 2]}),
        "test": pd.DataFrame({"id": ["c"], "val": [3]}),
    }
    audit = verify_no_leakage("demo", splits, id_col="id", strict=True)

    manifest = save_dataset_splits(
        dataset_name="demo_ds",
        splits=splits,
        output_dir=tmp_path,
        source="unit_test",
        scale="mini",
        preprocessing_profile="test_profile",
        leakage_audit=audit,
    )

    loaded_manifest = load_manifest(tmp_path / "demo_ds" / "manifest.json")
    assert loaded_manifest.split_counts == {"train": 2, "test": 1}
    assert loaded_manifest.sha256_checksums["train"] == manifest.sha256_checksums["train"]

    loaded_train = load_parquet(tmp_path / "demo_ds" / "train.parquet")
    assert list(loaded_train["id"]) == ["a", "b"]
