import chess

from chess_llm.core.board import (
    canonical_fen_key,
    is_legal_move,
    legal_moves,
    variant_fen_key,
    validate_fen,
)


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
OPEN_CASTLING_FEN = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
IMPOSSIBLE_BOARD_FEN = "8/8/8/8/8/8/4P3/4K3 w - - 0 1"


def test_canonical_fen_key_ignores_move_counters():
    fen_a = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"
    fen_b = "8/8/8/8/8/8/4K3/4k3 w - - 34 98"

    assert canonical_fen_key(fen_a) == canonical_fen_key(fen_b)
    assert canonical_fen_key(fen_a) == "8/8/8/8/8/8/4K3/4k3 w - -"


def test_canonical_fen_key_normalizes_non_legal_en_passant_square():
    fen_with_impossible_ep = (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq d3 0 1"
    )
    fen_without_ep = (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
    )

    assert canonical_fen_key(fen_with_impossible_ep) == canonical_fen_key(fen_without_ep)


def test_canonical_fen_key_preserves_chess960_castling_rights():
    board = chess.Board.from_chess960_pos(0)
    board.chess960 = True
    fen = board.fen()

    key = canonical_fen_key(fen, chess960=True)

    assert key == canonical_fen_key(fen)
    assert key.split()[2] == "KQkq"


def test_variant_fen_key_separates_standard_and_chess960_identity():
    fen = STARTING_FEN

    assert variant_fen_key(fen) == f"std:{canonical_fen_key(fen)}"
    assert variant_fen_key(fen, chess960=True) == (
        f"960:{canonical_fen_key(fen, chess960=True)}"
    )
    assert variant_fen_key(fen) != variant_fen_key(fen, chess960=True)


def test_validate_fen_returns_false_for_invalid_fen():
    assert validate_fen(STARTING_FEN)
    assert not validate_fen("not a fen")
    assert not validate_fen("8/8/8/8/8/8/8/8 w - - 0 1")


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
    assert not is_legal_move(IMPOSSIBLE_BOARD_FEN, "e2e4")
    assert not is_legal_move(STARTING_FEN, "not-a-move")


def test_is_legal_move_requires_canonical_generated_uci():
    assert "e1g1" in legal_moves(OPEN_CASTLING_FEN)
    assert is_legal_move(OPEN_CASTLING_FEN, "e1g1")
    assert not is_legal_move(OPEN_CASTLING_FEN, "e1h1")
