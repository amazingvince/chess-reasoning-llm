import chess

from chess_llm.core.opening_books import (
    OpeningBook,
    choose_weighted_move,
    discover_opening_books,
    sample_book_line,
)


def test_discover_opening_books_prefers_curated_order_and_falls_back(tmp_path):
    for name in ["Other.bin", "Titans.bin", "Human.bin"]:
        (tmp_path / name).write_bytes(b"book")

    books = discover_opening_books(tmp_path, curated_names=["Human.bin", "Missing.bin", "Titans.bin"])

    assert books == [
        OpeningBook(book_id="Human", name="Human", path=tmp_path / "Human.bin", curated=True),
        OpeningBook(book_id="Titans", name="Titans", path=tmp_path / "Titans.bin", curated=True),
        OpeningBook(book_id="Other", name="Other", path=tmp_path / "Other.bin", curated=False),
    ]


def test_choose_weighted_move_is_deterministic_for_fixed_seed():
    moves = [("e2e4", 70), ("d2d4", 20), ("g1f3", 10)]

    first = choose_weighted_move(moves, seed=7)
    second = choose_weighted_move(moves, seed=7)

    assert first == second
    assert first in {"e2e4", "d2d4", "g1f3"}


def test_sample_book_line_uses_weighted_provider_until_cap_or_missing_position():
    def provider(board: chess.Board) -> list[tuple[str, int]]:
        if board.fullmove_number == 1 and board.turn == chess.WHITE:
            return [("e2e4", 1)]
        if board.fullmove_number == 1 and board.turn == chess.BLACK:
            return [("e7e5", 1)]
        return []

    line = sample_book_line(provider, max_plies=6, seed=1)

    assert [move.move_uci for move in line.moves] == ["e2e4", "e7e5"]
    assert line.start_fen == chess.STARTING_FEN
    assert line.final_fen == "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
