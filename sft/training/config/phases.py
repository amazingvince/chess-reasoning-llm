"""Compatibility wrapper for package-owned training phase configuration."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.training.phases import (
        BASE_MODEL,
        DEFAULT_BASE_MODEL,
        PHASE_A,
        PHASE_B,
        PHASE_C,
        PHASE_PASSED_SENTINEL,
        PHASE_READY_SENTINEL,
        PHASES,
        PhaseConfig,
        TierMix,
        resolve_checkpoint,
    )
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.training.phases import (
        BASE_MODEL,
        DEFAULT_BASE_MODEL,
        PHASE_A,
        PHASE_B,
        PHASE_C,
        PHASE_PASSED_SENTINEL,
        PHASE_READY_SENTINEL,
        PHASES,
        PhaseConfig,
        TierMix,
        resolve_checkpoint,
    )

__all__ = [
    "BASE_MODEL",
    "DEFAULT_BASE_MODEL",
    "PHASE_A",
    "PHASE_B",
    "PHASE_C",
    "PHASE_PASSED_SENTINEL",
    "PHASE_READY_SENTINEL",
    "PHASES",
    "PhaseConfig",
    "TierMix",
    "resolve_checkpoint",
]
