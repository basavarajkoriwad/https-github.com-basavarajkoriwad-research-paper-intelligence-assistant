"""Configure local workspace Hugging Face cache before any test module imports transformers."""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_HF_CACHE = _ROOT / "artifacts" / "hf_cache"
if _HF_CACHE.exists():
    os.environ.setdefault("HF_HOME", str(_HF_CACHE))
    os.environ.setdefault("HF_HUB_CACHE", str(_HF_CACHE / "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
