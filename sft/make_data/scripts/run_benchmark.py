#!/usr/bin/env python3
"""Compatibility wrapper for ``chess_llm.evals.run_benchmark``."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.evals.run_benchmark import (
        _ACPL_INVALID_MOVE_PENALTY,
        _MOVE_TASK_TYPES,
        _evaluate_predicted_move,
        _example_is_chess960,
        _white_cp_to_side_to_move_cp,
        compute_acpl,
        evaluate_predicted_move,
        example_is_chess960,
        load_predictions,
        main,
        print_report,
        white_cp_to_side_to_move_cp,
    )
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.evals.run_benchmark import (
        _ACPL_INVALID_MOVE_PENALTY,
        _MOVE_TASK_TYPES,
        _evaluate_predicted_move,
        _example_is_chess960,
        _white_cp_to_side_to_move_cp,
        compute_acpl,
        evaluate_predicted_move,
        example_is_chess960,
        load_predictions,
        main,
        print_report,
        white_cp_to_side_to_move_cp,
    )

__all__ = [
    "_ACPL_INVALID_MOVE_PENALTY",
    "_MOVE_TASK_TYPES",
    "_evaluate_predicted_move",
    "_example_is_chess960",
    "_white_cp_to_side_to_move_cp",
    "compute_acpl",
    "evaluate_predicted_move",
    "example_is_chess960",
    "load_predictions",
    "main",
    "print_report",
    "white_cp_to_side_to_move_cp",
]


if __name__ == "__main__":
    raise SystemExit(main())
