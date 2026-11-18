"""Configuration loader supporting base YAML + hardware profile overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def find_project_root(start_path: Path | None = None) -> Path:
    """Locate the repository root containing pyproject.toml or configs/."""
    current = (start_path or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / "configs").exists():
            return candidate
    return Path.cwd().resolve()


class DatasetScaleConfig(BaseModel):
    """Row count caps for manageable sampling and full-scale runs."""

    arxiv_max_records: int | None = 5000
    squad_v2_train: int | None = 3000
    squad_v2_val: int | None = 800
    arxiv_summ_train: int | None = 500
    arxiv_summ_val: int | None = 100
    arxiv_summ_test: int | None = 100
    sciq_train: int | None = 2000
    sciq_val: int | None = 500
    sciq_test: int | None = 500


class ChunkingConfig(BaseModel):
    """Configuration for scientific document chunking."""

    strategy: str = "hierarchical_section"
    max_tokens: int = 384
    min_tokens: int = 64
    overlap_sentences: int = 2
    parent_max_tokens: int = 1500
    include_breadcrumb: bool = True


class AppConfig(BaseModel):
    """Typed application configuration."""

    project_root: Path
    profile: str = "cpu_lite"
    seed: int = 42
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path
    artifacts_dir: Path
    default_scale: str = "small"
    scales: dict[str, DatasetScaleConfig] = Field(default_factory=dict)
    arxiv_kaggle_url: str = "https://www.kaggle.com/datasets/Cornell-University/arxiv"
    arxiv_manual_json_path: Path
    arxiv_target_categories: list[str] = Field(default_factory=list)
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    spacy_model: str = "en_core_web_sm"
    preserve_negations: bool = True
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    hardware: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)

    def get_scale(self, scale_name: str | None = None) -> DatasetScaleConfig:
        """Return scale limits by name (mini, small, medium, full)."""
        name = scale_name or self.default_scale
        if name not in self.scales:
            valid = ", ".join(sorted(self.scales.keys()))
            raise ValueError(f"Unknown dataset scale '{name}'. Valid options: {valid}")
        return self.scales[name]

    def ensure_directories(self) -> None:
        """Create required data and artifact directories if missing."""
        for directory in [
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.reports_dir,
            self.artifacts_dir,
            self.arxiv_manual_json_path.parent,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (root / path).resolve()


def load_config(
    profile: str | None = None,
    project_root: Path | None = None,
) -> AppConfig:
    """Load base YAML config merged with the selected hardware profile."""
    root = find_project_root(project_root)
    base_yaml_path = root / "configs" / "base.yaml"
    with base_yaml_path.open("r", encoding="utf-8") as fh:
        raw_cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    selected_profile = profile or os.getenv("SCHOLARMIND_PROFILE", "cpu_lite")
    profile_yaml_path = root / "configs" / "profiles" / f"{selected_profile}.yaml"
    if profile_yaml_path.exists():
        with profile_yaml_path.open("r", encoding="utf-8") as fh:
            profile_cfg: dict[str, Any] = yaml.safe_load(fh) or {}
        raw_cfg = _deep_merge(raw_cfg, profile_cfg)

    paths_cfg = raw_cfg.get("paths", {})
    datasets_cfg = raw_cfg.get("datasets", {})
    arxiv_cfg = datasets_cfg.get("arxiv", {})
    splits_cfg = datasets_cfg.get("splits", {})
    nlp_cfg = raw_cfg.get("nlp", {})

    data_dir = _resolve_path(root, os.getenv("SCHOLARMIND_DATA_DIR", paths_cfg.get("data_dir", "data")))
    raw_dir = _resolve_path(root, paths_cfg.get("raw_dir", str(data_dir / "raw")))
    processed_dir = _resolve_path(root, paths_cfg.get("processed_dir", str(data_dir / "processed")))
    reports_dir = _resolve_path(root, paths_cfg.get("reports_dir", str(data_dir / "reports")))
    artifacts_dir = _resolve_path(
        root, os.getenv("SCHOLARMIND_ARTIFACTS_DIR", paths_cfg.get("artifacts_dir", "artifacts"))
    )
    manual_arxiv = _resolve_path(
        root,
        os.getenv(
            "ARXIV_KAGGLE_JSON_PATH",
            arxiv_cfg.get("manual_json_path", "data/raw/arxiv/arxiv-metadata-oai-snapshot.json"),
        ),
    )

    scales = {
        name: DatasetScaleConfig(**vals)
        for name, vals in datasets_cfg.get("scales", {}).items()
    }

    return AppConfig(
        project_root=root,
        profile=selected_profile,
        seed=int(raw_cfg.get("project", {}).get("seed", 42)),
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        artifacts_dir=artifacts_dir,
        default_scale=datasets_cfg.get("default_scale", "small"),
        scales=scales,
        arxiv_kaggle_url=arxiv_cfg.get(
            "kaggle_url", "https://www.kaggle.com/datasets/Cornell-University/arxiv"
        ),
        arxiv_manual_json_path=manual_arxiv,
        arxiv_target_categories=arxiv_cfg.get("target_categories", []),
        train_ratio=float(splits_cfg.get("train_ratio", 0.80)),
        val_ratio=float(splits_cfg.get("val_ratio", 0.10)),
        test_ratio=float(splits_cfg.get("test_ratio", 0.10)),
        spacy_model=nlp_cfg.get("spacy_model", "en_core_web_sm"),
        preserve_negations=bool(nlp_cfg.get("preserve_negations", True)),
        chunking=ChunkingConfig(**nlp_cfg.get("chunking", {})),
        hardware=raw_cfg.get("hardware", {}),
        models=raw_cfg.get("models", {}),
    )
