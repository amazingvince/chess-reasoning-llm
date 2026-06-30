"""Compatibility wrapper for ``chess_llm.sft.sources.chess960``."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.sft.sources.chess960 import (
        apply_random_moves,
        generate_all,
        generate_random,
        sample_chess960_positions,
    )
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.sft.sources.chess960 import (
        apply_random_moves,
        generate_all,
        generate_random,
        sample_chess960_positions,
    )

__all__ = [
    "apply_random_moves",
    "generate_all",
    "generate_random",
    "sample_chess960_positions",
]
