"""Tests for sources/syzygy_probing.py — DTZ best-move logic, material parsing (~8 cases)."""

import chess
import pytest

from conftest import KRK_FEN
from sources.syzygy_probing import (
    _material_signature,
    _parse_material,
    best_dtz_move,
)


def _build_probe_map(board, move_results):
    """Helper: push each move, record FEN -> result, pop.

    *move_results* is a list of (move, result_dict_or_None).
    Returns a dict mapping resulting FEN -> probe result.
    """
    probe_map = {}
    for move, result in move_results:
        board.push(move)
        probe_map[board.fen()] = result
        board.pop()
    return probe_map


def test_prefers_fastest_win(monkeypatch):
    """Two winning moves (DTZ +3 vs +7) — pick fastest (smallest positive)."""
    board = chess.Board(KRK_FEN)
    moves = list(board.legal_moves)
    assert len(moves) >= 2

    # our_dtz = -result["dtz"]; positive = winning
    probe_map = _build_probe_map(board, [
        (moves[0], {"wdl": -2, "dtz": -3}),   # our_dtz = +3
        (moves[1], {"wdl": -2, "dtz": -7}),   # our_dtz = +7
    ])

    monkeypatch.setattr(
        "sources.syzygy_probing.probe",
        lambda tb, b: probe_map.get(b.fen()),
    )
    assert best_dtz_move(None, board) == moves[0].uci()


def test_prefers_win_over_draw(monkeypatch):
    board = chess.Board(KRK_FEN)
    moves = list(board.legal_moves)

    probe_map = _build_probe_map(board, [
        (moves[0], {"wdl": -2, "dtz": -5}),   # our_dtz = +5 (win)
        (moves[1], {"wdl": 0, "dtz": 0}),     # draw
    ])

    monkeypatch.setattr(
        "sources.syzygy_probing.probe",
        lambda tb, b: probe_map.get(b.fen()),
    )
    assert best_dtz_move(None, board) == moves[0].uci()


def test_prefers_draw_over_loss(monkeypatch):
    board = chess.Board(KRK_FEN)
    moves = list(board.legal_moves)

    probe_map = _build_probe_map(board, [
        (moves[0], {"wdl": 0, "dtz": 0}),     # draw
        (moves[1], {"wdl": 2, "dtz": 5}),     # our_dtz = -5 (loss)
    ])

    monkeypatch.setattr(
        "sources.syzygy_probing.probe",
        lambda tb, b: probe_map.get(b.fen()),
    )
    assert best_dtz_move(None, board) == moves[0].uci()


def test_slowest_loss(monkeypatch):
    """Two losses (DTZ -3 and -10) — pick slowest (most negative)."""
    board = chess.Board(KRK_FEN)
    moves = list(board.legal_moves)

    probe_map = _build_probe_map(board, [
        (moves[0], {"wdl": 2, "dtz": 3}),     # our_dtz = -3 (fast loss)
        (moves[1], {"wdl": 2, "dtz": 10}),    # our_dtz = -10 (slow loss)
    ])

    monkeypatch.setattr(
        "sources.syzygy_probing.probe",
        lambda tb, b: probe_map.get(b.fen()),
    )
    assert best_dtz_move(None, board) == moves[1].uci()


def test_no_probes_returns_none(monkeypatch):
    board = chess.Board(KRK_FEN)
    monkeypatch.setattr(
        "sources.syzygy_probing.probe",
        lambda tb, b: None,
    )
    assert best_dtz_move(None, board) is None


def test_parse_material_kqk():
    white, black = _parse_material("KQK")
    assert set(white) == {chess.KING, chess.QUEEN}
    assert set(black) == {chess.KING}


def test_parse_material_krpkr():
    white, black = _parse_material("KRPKR")
    assert set(white) == {chess.KING, chess.ROOK, chess.PAWN}
    assert set(black) == {chess.KING, chess.ROOK}


def test_material_signature():
    board = chess.Board(KRK_FEN)
    assert _material_signature(board) == "KRK"
