"""Launch the ScholarMind AI Streamlit Research Paper RAG Assistant."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    hf_cache = root / "artifacts" / "hf_cache"
    env = dict(os.environ)
    if hf_cache.exists() and "HF_HOME" not in env:
        env["HF_HOME"] = str(hf_cache)
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")

    app_path = root / "src" / "scholarmind" / "ui" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
