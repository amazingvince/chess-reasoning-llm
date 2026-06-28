from chess_llm.core.board import (
    canonical_fen_key,
    is_legal_move,
    legal_moves,
    validate_fen,
)


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_canonical_fen_key_ignores_move_counters():
    fen_a = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"
    fen_b = "8/8/8/8/8/8/4K3/4k3 w - - 34 98"

    assert canonical_fen_key(fen_a) == canonical_fen_key(fen_b)
    assert canonical_fen_key(fen_a) == "8/8/8/8/8/8/4K3/4k3 w - -"


def test_validate_fen_returns_false_for_invalid_fen():
    assert validate_fen(STARTING_FEN)
    assert not validate_fen("not a fen")


def test_legal_moves_returns_sorted_uci_moves_and_empty_for_invalid_fen():
    moves = legal_moves(STARTING_FEN)

    assert moves == sorted(moves)
    assert "e2e4" in moves
    assert "g1f3" in moves
    assert legal_moves("not a fen") == []


def test_is_legal_move_returns_false_for_illegal_or_invalid_inputs():
    assert is_legal_move(STARTING_FEN, "e2e4")
    assert not is_legal_move(STARTING_FEN, "e2e5")
    assert not is_legal_move("not a fen", "e2e4")
    assert not is_legal_move(STARTING_FEN, "not-a-move")
