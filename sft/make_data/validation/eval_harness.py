"""Compatibility wrapper for package-owned eval harness APIs."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.evals.eval_harness import (
        SPLIT_CHECKS,
        EvalResult,
        answer_best_move_exists,
        answer_captures,
        answer_check_detection,
        answer_endgame_classification,
        answer_endgame_wdl,
        answer_legal_moves,
        answer_material_balance,
        answer_mate_choice,
        answer_opening_name,
        answer_pawn_structure,
        answer_position_eval,
        evaluate_split,
    )
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.evals.eval_harness import (
        SPLIT_CHECKS,
        EvalResult,
        answer_best_move_exists,
        answer_captures,
        answer_check_detection,
        answer_endgame_classification,
        answer_endgame_wdl,
        answer_legal_moves,
        answer_material_balance,
        answer_mate_choice,
        answer_opening_name,
        answer_pawn_structure,
        answer_position_eval,
        evaluate_split,
    )

__all__ = [
    "SPLIT_CHECKS",
    "EvalResult",
    "answer_best_move_exists",
    "answer_captures",
    "answer_check_detection",
    "answer_endgame_classification",
    "answer_endgame_wdl",
    "answer_legal_moves",
    "answer_material_balance",
    "answer_mate_choice",
    "answer_opening_name",
    "answer_pawn_structure",
    "answer_position_eval",
    "evaluate_split",
]
