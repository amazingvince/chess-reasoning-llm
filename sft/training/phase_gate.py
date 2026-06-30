"""Compatibility wrapper for package-owned phase gate criteria."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.training.phase_gate import check_phase_criteria
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.training.phase_gate import check_phase_criteria

__all__ = ["check_phase_criteria"]
