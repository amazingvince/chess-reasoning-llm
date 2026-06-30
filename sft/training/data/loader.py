"""Compatibility wrapper for package-owned training data loading."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.training.data.loader import (
        _KEEP_COLS,
        _load_sanitized_jsonl,
        _sanitized_jsonl_path,
        _write_sanitized_jsonl,
        load_tier_data,
    )
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.training.data.loader import (
        _KEEP_COLS,
        _load_sanitized_jsonl,
        _sanitized_jsonl_path,
        _write_sanitized_jsonl,
        load_tier_data,
    )

__all__ = [
    "_KEEP_COLS",
    "_load_sanitized_jsonl",
    "_sanitized_jsonl_path",
    "_write_sanitized_jsonl",
    "load_tier_data",
]
