"""Leakage-free deduplication, group-aware splitting, and cross-split leakage verification."""

from __future__ import annotations

import hashlib
import re
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from scholarmind.core.logging import get_logger
from scholarmind.core.schemas import LeakageAuditReport

logger = get_logger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")


def compute_text_hash(text: str | None) -> str:
    """Compute a normalized SHA-1 content hash for exact and near-exact duplicate detection."""
    if not text or not isinstance(text, str):
        return ""
    norm = _NON_ALNUM_RE.sub(" ", text.lower())
    norm = _WHITESPACE_RE.sub(" ", norm).strip()
    if not norm:
        return ""
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def deduplicate_dataframe(
    df: pd.DataFrame,
    text_cols: list[str],
    id_col: str | None = None,
) -> tuple[pd.DataFrame, int]:
    """Deduplicate records BEFORE splitting to prevent duplicate leakage across splits."""
    initial_count = len(df)
    working = df.copy()

    if id_col and id_col in working.columns:
        working = working.drop_duplicates(subset=[id_col], keep="first")

    if text_cols:
        combined_hash = working[text_cols].fillna("").astype(str).agg(" || ".join, axis=1).map(
            compute_text_hash
        )
        working = working.assign(_dedup_hash=combined_hash)
        # Keep rows with unique non-empty hashes
        non_empty = working["_dedup_hash"] != ""
        dup_mask = non_empty & working.duplicated(subset=["_dedup_hash"], keep="first")
        working = working.loc[~dup_mask].drop(columns=["_dedup_hash"])

    removed = initial_count - len(working)
    if removed > 0:
        logger.info("Deduplicated %d duplicate rows (%d -> %d).", removed, initial_count, len(working))
    return working.reset_index(drop=True), removed


def _assign_group_split_indices(
    df: pd.DataFrame,
    group_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition indices into train/val/test strictly respecting group boundaries."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-5):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    groups = df[group_col].astype(str).values
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < 3:
        raise ValueError(
            f"Need at least 3 unique groups in '{group_col}' for train/val/test split; found {n_groups}."
        )

    holdout_ratio = val_ratio + test_ratio
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=holdout_ratio, random_state=seed)
    train_idx, holdout_idx = next(gss_outer.split(df, groups=groups))

    holdout_groups = groups[holdout_idx]
    if len(np.unique(holdout_groups)) < 2:
        # Fallback: split unique groups deterministically if small sample caused 1 holdout group
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(unique_groups)
        n_train = max(1, int(round(n_groups * train_ratio)))
        n_val = max(1, int(round(n_groups * val_ratio)))
        train_g = set(shuffled[:n_train])
        val_g = set(shuffled[n_train : n_train + n_val])
        test_g = set(shuffled[n_train + n_val :])
        train_idx = np.where(np.isin(groups, list(train_g)))[0]
        val_idx = np.where(np.isin(groups, list(val_g)))[0]
        test_idx = np.where(np.isin(groups, list(test_g)))[0]
        return train_idx, val_idx, test_idx

    relative_test = test_ratio / holdout_ratio
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed + 1)
    val_rel, test_rel = next(gss_inner.split(holdout_idx, groups=holdout_groups))

    val_idx = holdout_idx[val_rel]
    test_idx = holdout_idx[test_rel]
    return train_idx, val_idx, test_idx


def group_aware_split(
    df: pd.DataFrame,
    group_col: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    stratify_col: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Create leakage-free train/val/test splits where no group crosses split boundaries.

    If `stratify_col` is provided and every group maps to a single label with sufficient
    group support (>= 3 groups per class), splits groups per stratum to preserve class balance
    while strictly keeping groups isolated.
    """
    if len(df) < 3:
        raise ValueError(f"Cannot split DataFrame with only {len(df)} rows.")

    working = df.reset_index(drop=True)

    if stratify_col and stratify_col in working.columns:
        # Check whether each group has a single primary label and enough groups per stratum
        group_labels = working.groupby(group_col)[stratify_col].first()
        label_group_counts = group_labels.value_counts()
        can_stratify = (label_group_counts >= 3).all() and len(label_group_counts) > 1
        if can_stratify:
            train_parts, val_parts, test_parts = [], [], []
            rng = np.random.default_rng(seed)
            for label, grp_series in group_labels.groupby(group_labels):
                lbl_groups = rng.permutation(grp_series.index.to_numpy())
                n_g = len(lbl_groups)
                n_val = max(1, int(round(n_g * val_ratio)))
                n_test = max(1, int(round(n_g * test_ratio)))
                n_train = max(1, n_g - n_val - n_test)
                tr_g = set(lbl_groups[:n_train])
                va_g = set(lbl_groups[n_train : n_train + n_val])
                te_g = set(lbl_groups[n_train + n_val :])
                train_parts.append(working[working[group_col].isin(tr_g)])
                val_parts.append(working[working[group_col].isin(va_g)])
                test_parts.append(working[working[group_col].isin(te_g)])
            return {
                "train": pd.concat(train_parts, ignore_index=True),
                "val": pd.concat(val_parts, ignore_index=True),
                "test": pd.concat(test_parts, ignore_index=True),
            }

    train_idx, val_idx, test_idx = _assign_group_split_indices(
        working,
        group_col=group_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    return {
        "train": working.iloc[train_idx].reset_index(drop=True),
        "val": working.iloc[val_idx].reset_index(drop=True),
        "test": working.iloc[test_idx].reset_index(drop=True),
    }


def remove_cross_split_leakage(
    splits: dict[str, pd.DataFrame],
    group_col: str | None = None,
    text_hash_col: str | None = None,
    priority_order: tuple[str, ...] = ("test", "val", "train"),
) -> tuple[dict[str, pd.DataFrame], int]:
    """Remove any accidental leakage across official pre-existing splits.

    Holds out higher-priority evaluation sets (`test`, then `val`) untouched and strips
    overlapping records from lower-priority splits (`train`) so evaluation remains unbiased.
    """
    cleaned = {k: v.copy() for k, v in splits.items()}
    seen_groups: set[str] = set()
    seen_hashes: set[str] = set()
    total_removed = 0

    for split_name in priority_order:
        if split_name not in cleaned or cleaned[split_name].empty:
            continue
        df = cleaned[split_name]
        keep_mask = pd.Series(True, index=df.index)

        if group_col and group_col in df.columns and seen_groups:
            g_vals = df[group_col].fillna("").astype(str)
            keep_mask &= ~((g_vals != "") & g_vals.isin(seen_groups))

        if text_hash_col and text_hash_col in df.columns and seen_hashes:
            h_vals = df[text_hash_col].fillna("").astype(str)
            keep_mask &= ~((h_vals != "") & h_vals.isin(seen_hashes))

        removed_here = int((~keep_mask).sum())
        if removed_here > 0:
            total_removed += removed_here
            df = df.loc[keep_mask].reset_index(drop=True)
            cleaned[split_name] = df
            logger.warning(
                "Removed %d cross-split leaking rows from '%s' to protect evaluation integrity.",
                removed_here,
                split_name,
            )

        if group_col and group_col in df.columns:
            seen_groups.update(x for x in df[group_col].fillna("").astype(str) if x)
        if text_hash_col and text_hash_col in df.columns:
            seen_hashes.update(x for x in df[text_hash_col].fillna("").astype(str) if x)

    return cleaned, total_removed


def verify_no_leakage(
    dataset_name: str,
    splits: dict[str, pd.DataFrame],
    id_col: str | None = None,
    group_col: str | None = None,
    text_hash_col: str | None = None,
    strict: bool = True,
) -> LeakageAuditReport:
    """Verify zero ID, group, and normalized text-hash overlap across all split pairs."""
    id_overlaps: dict[str, int] = {}
    group_overlaps: dict[str, int] = {}
    text_hash_overlaps: dict[str, int] = {}
    notes: list[str] = []

    active_splits = {k: v for k, v in splits.items() if v is not None and not v.empty}
    for s1, s2 in combinations(sorted(active_splits.keys()), 2):
        pair_key = f"{s1}_vs_{s2}"
        df1, df2 = active_splits[s1], active_splits[s2]

        if id_col and id_col in df1.columns and id_col in df2.columns:
            set1 = {x for x in df1[id_col].dropna().astype(str) if x}
            set2 = {x for x in df2[id_col].dropna().astype(str) if x}
            id_overlaps[pair_key] = len(set1 & set2)

        if group_col and group_col in df1.columns and group_col in df2.columns:
            g1 = {x for x in df1[group_col].dropna().astype(str) if x}
            g2 = {x for x in df2[group_col].dropna().astype(str) if x}
            group_overlaps[pair_key] = len(g1 & g2)

        if text_hash_col and text_hash_col in df1.columns and text_hash_col in df2.columns:
            h1 = {x for x in df1[text_hash_col].dropna().astype(str) if x}
            h2 = {x for x in df2[text_hash_col].dropna().astype(str) if x}
            text_hash_overlaps[pair_key] = len(h1 & h2)

    passed = (
        all(v == 0 for v in id_overlaps.values())
        and all(v == 0 for v in group_overlaps.values())
        and all(v == 0 for v in text_hash_overlaps.values())
    )

    if passed:
        notes.append("Verified 0 ID overlaps, 0 group overlaps, and 0 text-hash overlaps across splits.")
    else:
        notes.append("LEAKAGE DETECTED: non-zero overlap found across dataset splits.")

    report = LeakageAuditReport(
        dataset_name=dataset_name,
        passed=passed,
        id_overlaps=id_overlaps,
        group_overlaps=group_overlaps,
        text_hash_overlaps=text_hash_overlaps,
        notes=notes,
    )

    if strict and not passed:
        raise AssertionError(
            f"Data leakage detected in '{dataset_name}': "
            f"id={id_overlaps}, group={group_overlaps}, text_hash={text_hash_overlaps}"
        )

    return report
