"""Parquet persistence and reproducible JSON manifest storage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scholarmind.core.logging import get_logger
from scholarmind.core.schemas import LeakageAuditReport, SplitManifest

logger = get_logger(__name__)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 digest of a file on disk."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Save a DataFrame cleanly as compressed Apache Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    logger.info("Saved %d rows -> %s (%.2f MB)", len(df), path, path.stat().st_size / (1024 * 1024))
    return path


def load_parquet(path: Path) -> pd.DataFrame:
    """Load an Apache Parquet dataset split."""
    if not path.exists():
        raise FileNotFoundError(f"Processed Parquet file not found: {path}")
    return pd.read_parquet(path, engine="pyarrow")


def save_dataset_splits(
    dataset_name: str,
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
    source: str,
    scale: str,
    preprocessing_profile: str,
    leakage_audit: LeakageAuditReport,
    seed: int = 42,
    notes: dict[str, Any] | None = None,
) -> SplitManifest:
    """Save all splits as Parquet files alongside a checksummed JSON manifest."""
    ds_dir = output_dir / dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    checksums: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    columns: list[str] = []

    for split_name, df in splits.items():
        if df is None:
            continue
        file_path = ds_dir / f"{split_name}.parquet"
        save_parquet(df, file_path)
        files[split_name] = file_path.name
        checksums[split_name] = sha256_file(file_path)
        split_counts[split_name] = int(len(df))
        if not columns and len(df.columns) > 0:
            columns = list(df.columns)

    manifest = SplitManifest(
        dataset_name=dataset_name,
        source=source,
        scale=scale,
        seed=seed,
        preprocessing_profile=preprocessing_profile,
        split_counts=split_counts,
        columns=columns,
        files=files,
        sha256_checksums=checksums,
        leakage_audit=leakage_audit,
        notes=notes or {},
    )

    manifest_path = ds_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Wrote split manifest -> %s", manifest_path)
    return manifest


def load_manifest(manifest_path: Path) -> SplitManifest:
    """Load and validate a SplitManifest from JSON."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return SplitManifest.model_validate(payload)
