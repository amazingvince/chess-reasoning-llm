"""Tests for tier 4 generators: MaterialBalance, PositionEvaluation, PawnStructure."""

from random import Random

import chess
import pytest

from conftest import (
    DOUBLED_PAWNS_FEN,
    ISOLATED_PAWN_FEN,
    KRK_FEN,
    PASSED_PAWN_FEN,
    QUEEN_GONE_FEN,
    STARTING_FEN,
)


# ── _cp_to_bucket unit tests ───────────────────────────────────────


def test_cp_to_bucket_equal_positive():
    from generators.tier4_evaluation import _cp_to_bucket

    result = _cp_to_bucket(30)
    assert result == "The position is equal."


def test_cp_to_bucket_slight_edge():
    from generators.tier4_evaluation import _cp_to_bucket

    result = _cp_to_bucket(100)
    assert result == "White has a slight edge."


def test_cp_to_bucket_black_clear_advantage():
    from generators.tier4_evaluation import _cp_to_bucket

    result = _cp_to_bucket(-250)
    assert result == "Black has a clear advantage."


def test_cp_to_bucket_winning():
    from generators.tier4_evaluation import _cp_to_bucket

    result = _cp_to_bucket(400)
    assert result == "White is winning."


def test_cp_to_bucket_zero_is_equal():
    from generators.tier4_evaluation import _cp_to_bucket

    result = _cp_to_bucket(0)
    assert result == "The position is equal."


# ── _analyze_pawn_structure unit tests ──────────────────────────────


def test_pawn_structure_starting_fen():
    from generators.tier4_evaluation import _analyze_pawn_structure

    board = chess.Board(STARTING_FEN)
    analysis = _analyze_pawn_structure(board)
    for color in ("white", "black"):
        assert analysis[color]["doubled"] == []
        assert analysis[color]["isolated"] == []
        assert analysis[color]["passed"] == []


def test_pawn_structure_doubled():
    from generators.tier4_evaluation import _analyze_pawn_structure

    board = chess.Board(DOUBLED_PAWNS_FEN)
    analysis = _analyze_pawn_structure(board)
    assert "e" in analysis["white"]["doubled"]


def test_pawn_structure_passed():
    from generators.tier4_evaluation import _analyze_pawn_structure

    board = chess.Board(PASSED_PAWN_FEN)
    analysis = _analyze_pawn_structure(board)
    assert "e3" in analysis["white"]["passed"]


def test_pawn_structure_isolated():
    from generators.tier4_evaluation import _analyze_pawn_structure

    board = chess.Board(ISOLATED_PAWN_FEN)
    analysis = _analyze_pawn_structure(board)
    assert "e4" in analysis["white"]["isolated"]


# ── MaterialBalance generator ───────────────────────────────────────


def test_material_balance_equal(monkeypatch):
    from generators.tier4_evaluation import MaterialBalance

    tpl = "FEN: {fen}\nWhat is the material balance?"
    monkeypatch.setattr(
        "generators.tier4_evaluation.select_template", lambda tid, rng: tpl
    )

    gen = MaterialBalance(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1

    answer = examples[0]["messages"][2]["content"]
    assert "Material is equal." in answer
    assert "39 pts" in answer


def test_material_balance_black_ahead(monkeypatch):
    from generators.tier4_evaluation import MaterialBalance

    tpl = "FEN: {fen}\nWhat is the material balance?"
    monkeypatch.setattr(
        "generators.tier4_evaluation.select_template", lambda tid, rng: tpl
    )

    gen = MaterialBalance(
        config={"fen_pool": [{"fen": QUEEN_GONE_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1

    answer = examples[0]["messages"][2]["content"]
    assert "Black is ahead by 9 point(s)." in answer


# ── PositionEvaluation generator ────────────────────────────────────


def test_position_eval_equal(monkeypatch):
    from generators.tier4_evaluation import PositionEvaluation

    tpl = "FEN: {fen}\nEvaluate this position. Who is better?"
    monkeypatch.setattr(
        "generators.tier4_evaluation.select_template", lambda tid, rng: tpl
    )

    gen = PositionEvaluation(
        config={
            "position_evals": [{"fen": STARTING_FEN, "cp": 30, "mate": None}],
            "volume_override": 1,
        },
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1
    assert "equal" in examples[0]["messages"][2]["content"].lower()


def test_position_eval_forced_mate(monkeypatch):
    from generators.tier4_evaluation import PositionEvaluation

    tpl = "FEN: {fen}\nEvaluate this position. Who is better?"
    monkeypatch.setattr(
        "generators.tier4_evaluation.select_template", lambda tid, rng: tpl
    )

    gen = PositionEvaluation(
        config={
            "position_evals": [{"fen": KRK_FEN, "cp": None, "mate": 3}],
            "volume_override": 1,
        },
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1
    assert "decisive advantage (forced mate)" in examples[0]["messages"][2]["content"]


# ── PawnStructure generator ─────────────────────────────────────────


def test_pawn_structure_gen_no_features(monkeypatch):
    from generators.tier4_evaluation import PawnStructure

    tpl = "FEN: {fen}\nAnalyze the pawn structure."
    monkeypatch.setattr(
        "generators.tier4_evaluation.select_template", lambda tid, rng: tpl
    )

    gen = PawnStructure(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1
    assert examples[0]["messages"][2]["content"] == "No notable pawn structure features."
