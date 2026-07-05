from pathlib import Path

import chess


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/8/8/8/8 w - - 0 1"
REAL_STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
CHESS960_CASTLE_FEN = "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1"


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
