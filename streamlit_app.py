"""Root entrypoint for 1-click Streamlit Community Cloud deployment."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from scholarmind.ui.app import main  # noqa: E402

if __name__ == "__main__":
    main()
