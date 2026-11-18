"""Launch the ScholarMind AI FastAPI backend server."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    hf_cache = root / "artifacts" / "hf_cache"
    if hf_cache.exists() and not os.getenv("HF_HOME"):
        os.environ["HF_HOME"] = str(hf_cache)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    uvicorn.run(
        "scholarmind.api.main:app",
        host=os.getenv("SCHOLARMIND_HOST", "127.0.0.1"),
        port=int(os.getenv("SCHOLARMIND_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
