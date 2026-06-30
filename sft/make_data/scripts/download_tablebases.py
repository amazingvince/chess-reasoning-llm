#!/usr/bin/env python3
"""Compatibility alias for package-owned Syzygy tablebase download."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.sft import download_tablebases as _download_tablebases
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.sft import download_tablebases as _download_tablebases

sys.modules[__name__] = _download_tablebases

if __name__ == "__main__":
    raise SystemExit(_download_tablebases.main())
