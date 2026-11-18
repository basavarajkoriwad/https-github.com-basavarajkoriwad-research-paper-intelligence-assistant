"""Prefetch CPU-friendly pretrained Hugging Face models into the local workspace cache."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from scholarmind.core.config import load_config
from scholarmind.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODELS = [
    "allenai/scibert_scivocab_uncased",
    "google/flan-t5-small",
    "google/flan-t5-base",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
]


def main() -> None:
    cfg = load_config()
    cache_dir = Path(os.getenv("HF_HOME", str(cfg.artifacts_dir / "hf_cache")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)

    for model_id in DEFAULT_MODELS:
        logger.info("Prefetching %s into %s...", model_id, cache_dir)
        snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir / "hub"),
            ignore_patterns=["*.h5", "*.ot", "*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        logger.info("Cached %s successfully.", model_id)


if __name__ == "__main__":
    main()
