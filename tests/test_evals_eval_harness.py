import importlib
import sys
from pathlib import Path

import chess


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/8/8/8/8 w - - 0 1"
REAL_STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
CHESS960_CASTLE_FEN = "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1"


def _clear_legacy_modules() -> None:
    for name in list(sys.modules):
        if (
            name.startswith("validation")
            or name.startswith("generators")
            or name == "config"
            or name.startswith("config.")
        ):
            sys.modules.pop(name, None)


def test_package_eval_harness_import_does_not_pull_legacy_modules():
    _clear_legacy_modules()

    from chess_llm.evals import eval_harness

    importlib.reload(eval_harness)

    assert "validation.eval_harness" not in sys.modules
    assert "validation.benchmark" not in sys.modules
    assert not any(name.startswith("generators.") for name in sys.modules)
    assert "config" not in sys.modules


def test_package_eval_harness_answers_and_invalid_fen_checks():
    from chess_llm.evals.eval_harness import (
        answer_captures,
        answer_legal_moves,
        evaluate_split,
    )

    expected = " ".join(sorted(move.uci() for move in chess.Board(REAL_STARTING_FEN).legal_moves))

    assert answer_legal_moves(REAL_STARTING_FEN) == expected
    assert answer_captures(REAL_STARTING_FEN) == "No captures available."

    result = evaluate_split("rules", [{"fen": STARTING_FEN}])

    assert result.total == 1
    assert result.failed == 1
    assert result.passed == 0
    assert any("invalid FEN" in error for error in result.errors)


def test_package_eval_harness_uses_chess960_id_metadata():
    from chess_llm.evals.eval_harness import evaluate_split

    result = evaluate_split(
        "mate",
        [{
            "fen": CHESS960_CASTLE_FEN,
            "chess960_id": 321,
            "move_a": "d1c1",
            "move_b": "d2d4",
            "better_move": "d1c1",
        }],
    )

    assert result.total == 1
    assert result.passed == 1
    assert result.failed == 0


def test_legacy_eval_harness_delegates_to_package():
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    sys.path.insert(0, str(make_data_root))

    from chess_llm.evals import eval_harness as packaged
    from validation import eval_harness as legacy

    assert legacy.EvalResult is packaged.EvalResult
    assert legacy.SPLIT_CHECKS is packaged.SPLIT_CHECKS
    assert legacy.answer_legal_moves is packaged.answer_legal_moves
    assert legacy.answer_captures is packaged.answer_captures
    assert legacy.evaluate_split is packaged.evaluate_split
