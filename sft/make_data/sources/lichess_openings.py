"""Compatibility wrapper for ``chess_llm.sft.sources.lichess_openings``."""

from __future__ import annotations

import sys
from pathlib import Path

from config.settings import HF_DATASETS as _HF_DATASETS  # applies legacy HF_HOME

try:
    from chess_llm.sft.sources.lichess_openings import load_openings, parse_opening_row
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.sft.sources.lichess_openings import load_openings, parse_opening_row

__all__ = ["load_openings", "parse_opening_row"]
