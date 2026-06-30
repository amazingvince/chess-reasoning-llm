"""Compatibility wrapper for package-owned training data mixing."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.training.data.mixer import (
        _fen_key,
        _select_eval_fen_keys,
        _split_by_fen,
        build_phase_dataset,
        summarize_phase_data,
    )
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.training.data.mixer import (
        _fen_key,
        _select_eval_fen_keys,
        _split_by_fen,
        build_phase_dataset,
        summarize_phase_data,
    )

__all__ = [
    "_fen_key",
    "_select_eval_fen_keys",
    "_split_by_fen",
    "build_phase_dataset",
    "summarize_phase_data",
]
