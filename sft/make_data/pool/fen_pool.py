"""Compatibility wrapper for ``chess_llm.sft.fen_pool``."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.sft.fen_pool import FENPool
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.sft.fen_pool import FENPool

__all__ = ["FENPool"]
