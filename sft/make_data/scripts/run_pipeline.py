#!/usr/bin/env python3
"""Compatibility alias for the package-owned SFT data pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.sft import pipeline as _pipeline
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.sft import pipeline as _pipeline

sys.modules[__name__] = _pipeline

if __name__ == "__main__":
    raise SystemExit(_pipeline.main())
