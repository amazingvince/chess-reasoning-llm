"""Tests for validation/eval_harness.py — answer functions and split evaluator."""

import json

import chess
import pytest

from conftest import CHECKMATE_FEN, KRK_FEN, STARTING_FEN


# ── answer_legal_moves ──────────────────────────────────────────────


def test_answer_legal_moves_starting():
    from validation.eval_harness import answer_legal_moves

    result = answer_legal_moves(STARTING_FEN)
    board = chess.Board(STARTING_FEN)
    expected = " ".join(sorted(m.uci() for m in board.legal_moves))
    assert result == expected


def test_answer_legal_moves_krk():
    from validation.eval_harness import answer_legal_moves

    result = answer_legal_moves(KRK_FEN)
    assert "e1" in result  # King moves from e1
    assert len(result.split()) > 0


# ── answer_check_detection ──────────────────────────────────────────


def test_answer_check_detection_normal():
    from validation.eval_harness import answer_check_detection

    assert answer_check_detection(STARTING_FEN) == "Normal"


def test_answer_check_detection_checkmate():
    from validation.eval_harness import answer_check_detection

    assert answer_check_detection(CHECKMATE_FEN) == "Checkmate"


# ── answer_captures ─────────────────────────────────────────────────


def test_answer_captures_starting():
    from validation.eval_harness import answer_captures

    # No captures available from starting position
    assert answer_captures(STARTING_FEN) == ""


def test_answer_captures_after_e4_d5():
    from validation.eval_harness import answer_captures

    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    result = answer_captures(fen)
    assert "e4d5" in result


# ── answer_position_eval ────────────────────────────────────────────


def test_answer_position_eval_equal():
    from validation.eval_harness import answer_position_eval

    result = answer_position_eval(cp=30, mate=None)
    assert "equal" in result.lower()


def test_answer_position_eval_mate():
    from validation.eval_harness import answer_position_eval

    result = answer_position_eval(cp=None, mate=3)
    assert "decisive advantage (forced mate)" in result


# ── answer_endgame_classification ───────────────────────────────────


def test_answer_endgame_classification_krk():
    from validation.eval_harness import answer_endgame_classification

    assert answer_endgame_classification(KRK_FEN) == "This is a KRK endgame."


# ── answer_endgame_wdl ──────────────────────────────────────────────


def test_answer_endgame_wdl_win():
    from validation.eval_harness import answer_endgame_wdl

    assert answer_endgame_wdl(2) == "Win for the side to move."


# ── evaluate_split ──────────────────────────────────────────────────


def test_evaluate_split_perception():
    from validation.eval_harness import evaluate_split

    examples = [
        {"fen": STARTING_FEN},
        {"fen": KRK_FEN},
    ]
    result = evaluate_split("perception", examples)
    assert result.total == 2
    assert result.passed == 2
    assert result.failed == 0


def test_evaluate_split_unknown_split():
    from validation.eval_harness import evaluate_split

    result = evaluate_split("nonexistent", [{"fen": STARTING_FEN}])
    assert result.skipped == 1


def test_evaluate_split_jsonl_roundtrip(tmp_path):
    """Verify JSONL loading works with evaluate_split."""
    from validation.eval_harness import evaluate_split

    data = [{"fen": STARTING_FEN}, {"fen": KRK_FEN}]
    path = tmp_path / "perception.jsonl"
    with open(path, "w") as fh:
        for d in data:
            fh.write(json.dumps(d) + "\n")

    loaded = []
    with open(path) as fh:
        for line in fh:
            loaded.append(json.loads(line.strip()))

    result = evaluate_split("perception", loaded)
    assert result.passed == 2


# ── New answer functions ──────────────────────────────────────────────


def test_answer_opening_name_valid():
    from validation.eval_harness import answer_opening_name

    result = answer_opening_name(STARTING_FEN, "Sicilian Defense", "B20")
    assert result == "Sicilian Defense (ECO: B20)"


def test_answer_best_move_exists_valid():
    from validation.eval_harness import answer_best_move_exists

    result = answer_best_move_exists(STARTING_FEN, "e2e4")
    assert result == "e2e4"


def test_answer_best_move_exists_illegal():
    from validation.eval_harness import answer_best_move_exists

    with pytest.raises(ValueError, match="not legal"):
        answer_best_move_exists(STARTING_FEN, "e1e5")


def test_answer_mate_choice_valid():
    from validation.eval_harness import answer_mate_choice

    result = answer_mate_choice(STARTING_FEN, "e2e4", "d2d4", "e2e4")
    assert result == "e2e4"


def test_answer_mate_choice_invalid_better():
    from validation.eval_harness import answer_mate_choice

    with pytest.raises(ValueError, match="not one of"):
        answer_mate_choice(STARTING_FEN, "e2e4", "d2d4", "a2a3")


# ── Split coverage ────────────────────────────────────────────────────


def test_all_nine_splits_have_checks():
    """All 9 splits now have entries in SPLIT_CHECKS."""
    from validation.eval_harness import SPLIT_CHECKS

    expected = {
        "perception", "rules", "tactics", "evaluation",
        "openings", "endgames", "planning", "chess960", "mate",
    }
    assert set(SPLIT_CHECKS.keys()) == expected


def test_perception_vs_rules_different_checks():
    """Perception and rules splits should have different check functions."""
    from validation.eval_harness import SPLIT_CHECKS

    perc = set(fn.__name__ for fn in SPLIT_CHECKS["perception"])
    rules = set(fn.__name__ for fn in SPLIT_CHECKS["rules"])
    assert perc != rules


def test_evaluate_planning_with_best_move_passes():
    """A planning row with best_move passes the check."""
    from validation.eval_harness import evaluate_split

    examples = [{"fen": STARTING_FEN, "best_move": "e2e4"}]
    result = evaluate_split("planning", examples)
    assert result.total == 1
    assert result.passed == 1
    assert result.skipped == 0


def test_evaluate_planning_no_move_at_all_is_skipped():
    """A planning row with no move fields at all is skipped, not passed."""
    from validation.eval_harness import evaluate_split

    examples = [{"fen": STARTING_FEN}]
    result = evaluate_split("planning", examples)
    assert result.total == 1
    assert result.skipped == 1
    assert result.passed == 0
