"""Tests for tier 6 generators: endgame classification, WDL, best move, principles."""

from random import Random

import chess
import pytest

from conftest import KRK_FEN, STARTING_FEN


# ── _material_signature unit tests ──────────────────────────────────


def test_material_signature_krk():
    from generators.tier6_endgames import _material_signature

    board = chess.Board(KRK_FEN)
    assert _material_signature(board) == "KRK"


def test_material_signature_starting():
    from generators.tier6_endgames import _material_signature

    board = chess.Board(STARTING_FEN)
    sig = _material_signature(board)
    # Starting position: KQRRBBNNPPPPPPPP for each side
    assert sig == "KQRRBBNNPPPPPPPPKQRRBBNNPPPPPPPP"


# ── _WDL_LABELS ─────────────────────────────────────────────────────


def test_wdl_labels_all_values():
    from generators.tier6_endgames import _WDL_LABELS

    for wdl in (2, 1, 0, -1, -2):
        assert wdl in _WDL_LABELS
        assert isinstance(_WDL_LABELS[wdl], str)


def test_wdl_labels_unknown_fallback():
    from generators.tier6_endgames import _WDL_LABELS

    # Unknown WDL value falls back to raw string via dict.get
    assert _WDL_LABELS.get(99, f"WDL value: {99}") == "WDL value: 99"


# ── EndgameClassification generator ─────────────────────────────────


def test_endgame_classification_krk(monkeypatch):
    from generators.tier6_endgames import EndgameClassification

    tpl = "FEN: {fen}\nWhat type of endgame is this?"
    monkeypatch.setattr(
        "generators.tier6_endgames.select_template", lambda tid, rng: tpl
    )

    eg = {"fen": KRK_FEN, "material": "KRK"}
    gen = EndgameClassification(
        config={"endgame_positions": [eg], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1
    assert "This is a KRK endgame." in examples[0]["messages"][2]["content"]


def test_endgame_classification_auto_signature(monkeypatch):
    """When material key is missing, signature is computed from FEN."""
    from generators.tier6_endgames import EndgameClassification

    tpl = "FEN: {fen}\nWhat type of endgame is this?"
    monkeypatch.setattr(
        "generators.tier6_endgames.select_template", lambda tid, rng: tpl
    )

    eg = {"fen": KRK_FEN}  # No material key
    gen = EndgameClassification(
        config={"endgame_positions": [eg], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1
    assert "This is a KRK endgame." in examples[0]["messages"][2]["content"]


# ── EndgameWDL generator ────────────────────────────────────────────


def test_endgame_wdl_win(monkeypatch):
    from generators.tier6_endgames import EndgameWDL

    tpl = "FEN: {fen}\nIs this endgame a win, draw, or loss for the side to move?"
    monkeypatch.setattr(
        "generators.tier6_endgames.select_template", lambda tid, rng: tpl
    )

    eg = {"fen": KRK_FEN, "wdl": 2}
    gen = EndgameWDL(
        config={"endgame_positions": [eg], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1
    assert "Win for the side to move." in examples[0]["messages"][2]["content"]


def test_endgame_wdl_draw(monkeypatch):
    from generators.tier6_endgames import EndgameWDL

    tpl = "FEN: {fen}\nIs this endgame a win, draw, or loss for the side to move?"
    monkeypatch.setattr(
        "generators.tier6_endgames.select_template", lambda tid, rng: tpl
    )

    eg = {"fen": KRK_FEN, "wdl": 0}
    gen = EndgameWDL(
        config={"endgame_positions": [eg], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1
    assert "Draw with best play." in examples[0]["messages"][2]["content"]


def test_endgame_wdl_none_skipped(monkeypatch):
    from generators.tier6_endgames import EndgameWDL

    tpl = "FEN: {fen}\nIs this endgame a win, draw, or loss for the side to move?"
    monkeypatch.setattr(
        "generators.tier6_endgames.select_template", lambda tid, rng: tpl
    )

    eg = {"fen": KRK_FEN}  # No wdl key
    gen = EndgameWDL(
        config={"endgame_positions": [eg], "volume_override": 1},
        rng=Random(42),
    )
    assert list(gen.generate()) == []


# ── EndgameBestMove generator ───────────────────────────────────────


def test_endgame_best_move(monkeypatch):
    from generators.tier6_endgames import EndgameBestMove

    tpl = "FEN: {fen}\nWhat is the best move in this endgame?"
    monkeypatch.setattr(
        "generators.tier6_endgames.select_template", lambda tid, rng: tpl
    )

    eg = {"fen": KRK_FEN, "best_move": "h1h7", "wdl": 2, "dtz": 10}
    gen = EndgameBestMove(
        config={"endgame_positions": [eg], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1
    assert examples[0]["messages"][2]["content"] == "h1h7"


def test_endgame_best_move_empty_skipped(monkeypatch):
    from generators.tier6_endgames import EndgameBestMove

    tpl = "FEN: {fen}\nWhat is the best move in this endgame?"
    monkeypatch.setattr(
        "generators.tier6_endgames.select_template", lambda tid, rng: tpl
    )

    eg = {"fen": KRK_FEN, "best_move": ""}
    gen = EndgameBestMove(
        config={"endgame_positions": [eg], "volume_override": 1},
        rng=Random(42),
    )
    assert list(gen.generate()) == []


# ── EndgamePrinciples generator ─────────────────────────────────────


def test_endgame_principles_kpk(monkeypatch):
    from generators.tier6_endgames import EndgamePrinciples, _ENDGAME_PRINCIPLES

    tpl = "FEN: {fen}\nWhat endgame principles apply here?"
    monkeypatch.setattr(
        "generators.tier6_endgames.select_template", lambda tid, rng: tpl
    )

    # KPK: King + Pawn vs King
    kpk_fen = "4k3/8/8/8/8/4P3/8/4K3 w - - 0 1"
    eg = {"fen": kpk_fen, "material": "KPK"}
    gen = EndgamePrinciples(
        config={"endgame_positions": [eg], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1

    answer = examples[0]["messages"][2]["content"]
    # Answer should be drawn from KPK principles
    assert any(p in answer for p in _ENDGAME_PRINCIPLES["KPK"])


def test_endgame_principles_unknown_material(monkeypatch):
    from generators.tier6_endgames import EndgamePrinciples

    tpl = "FEN: {fen}\nWhat endgame principles apply here?"
    monkeypatch.setattr(
        "generators.tier6_endgames.select_template", lambda tid, rng: tpl
    )

    eg = {"fen": KRK_FEN, "material": "XYZZY"}
    gen = EndgamePrinciples(
        config={"endgame_positions": [eg], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1

    answer = examples[0]["messages"][2]["content"]
    # KRK has 3 pieces (<=4), so generic few-pieces advice
    assert "king activity" in answer.lower() or "centralize" in answer.lower()
