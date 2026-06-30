#!/usr/bin/env python3
"""Compatibility wrapper for ``chess_llm.evals.freeze_benchmark``."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.evals.freeze_benchmark import load_raw_split, main
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.evals.freeze_benchmark import load_raw_split, main

__all__ = ["load_raw_split", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
