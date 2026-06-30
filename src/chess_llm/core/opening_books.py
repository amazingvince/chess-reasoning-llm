"""Polyglot opening-book helpers for UI and evaluation consumers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import random

import chess
import chess.polyglot


DEFAULT_CURATED_BOOKS = ("Human.bin", "Titans.bin", "baron30.bin", "komodo.bin")


@dataclass(frozen=True)
class OpeningBook:
    """One discovered Polyglot book file."""

    book_id: str
    name: str
    path: Path
    curated: bool = False


@dataclass(frozen=True)
class SampledBookMove:
    """One move sampled from an opening book."""

    ply: int
    fen_before: str
    move_uci: str
    weight: int | None = None


@dataclass(frozen=True)
class SampledBookLine:
    """A sampled opening line from a starting position."""

    start_fen: str
    final_fen: str
    moves: list[SampledBookMove]


WeightedMoveProvider = Callable[[chess.Board], Sequence[tuple[str, int]]]


def discover_opening_books(
    root: str | Path,
    curated_names: Sequence[str] = DEFAULT_CURATED_BOOKS,
) -> list[OpeningBook]:
    """Return discovered ``.bin`` books, with curated books first."""
    root_path = Path(root)
    if not root_path.exists():
        return []

    by_name = {path.name: path for path in root_path.glob("*.bin") if path.is_file()}
    books: list[OpeningBook] = []
    seen: set[str] = set()

    for name in curated_names:
        path = by_name.get(name)
        if path is None:
            continue
        books.append(_book_from_path(path, curated=True))
        seen.add(path.name)

    for path in sorted(by_name.values(), key=lambda item: item.name.lower()):
        if path.name in seen:
            continue
        books.append(_book_from_path(path, curated=False))

    return books


def choose_weighted_move(
    moves: Sequence[tuple[str, int]],
    *,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> str | None:
    """Choose one move from ``[(uci, weight), ...]`` using weighted sampling."""
    legal_weighted = [(move, int(weight)) for move, weight in moves if int(weight) > 0]
    if not legal_weighted:
        return None

    chooser = rng if rng is not None else random.Random(seed)
    total = sum(weight for _, weight in legal_weighted)
    target = chooser.uniform(0, total)
    cumulative = 0.0
    for move, weight in legal_weighted:
        cumulative += weight
        if target <= cumulative:
            return move
    return legal_weighted[-1][0]


def get_weighted_book_moves(
    book_path: str | Path,
    board: chess.Board,
) -> list[tuple[str, int]]:
    """Read weighted legal moves for ``board`` from a Polyglot book."""
    try:
        with chess.polyglot.open_reader(str(book_path)) as reader:
            entries = list(reader.find_all(board))
    except (FileNotFoundError, KeyError):
        return []

    moves: list[tuple[str, int]] = []
    for entry in entries:
        if entry.move in board.legal_moves:
            moves.append((entry.move.uci(), int(entry.weight)))
    moves.sort(key=lambda item: item[1], reverse=True)
    return moves


def sample_book_line(
    provider: WeightedMoveProvider,
    *,
    max_plies: int,
    seed: int | None = None,
    start_fen: str = chess.STARTING_FEN,
    chess960: bool = False,
) -> SampledBookLine:
    """Sample a weighted opening line from a move provider."""
    board = chess.Board(start_fen, chess960=chess960)
    rng = random.Random(seed)
    sampled: list[SampledBookMove] = []

    for ply in range(max(0, int(max_plies))):
        weighted_moves = list(provider(board))
        move_uci = choose_weighted_move(weighted_moves, rng=rng)
        if move_uci is None:
            break

        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            break
        if move not in board.legal_moves:
            break

        weight = _weight_for_move(weighted_moves, move_uci)
        sampled.append(
            SampledBookMove(
                ply=ply,
                fen_before=board.fen(),
                move_uci=move_uci,
                weight=weight,
            )
        )
        board.push(move)

    return SampledBookLine(
        start_fen=start_fen,
        final_fen=board.fen(),
        moves=sampled,
    )


def sample_polyglot_book_line(
    book_path: str | Path,
    *,
    max_plies: int,
    seed: int | None = None,
    start_fen: str = chess.STARTING_FEN,
    chess960: bool = False,
) -> SampledBookLine:
    """Sample a line directly from a Polyglot ``.bin`` file."""
    return sample_book_line(
        lambda board: get_weighted_book_moves(book_path, board),
        max_plies=max_plies,
        seed=seed,
        start_fen=start_fen,
        chess960=chess960,
    )


def _book_from_path(path: Path, *, curated: bool) -> OpeningBook:
    stem = path.stem
    return OpeningBook(book_id=stem, name=stem, path=path, curated=curated)


def _weight_for_move(moves: Sequence[tuple[str, int]], move_uci: str) -> int | None:
    for move, weight in moves:
        if move == move_uci:
            return int(weight)
    return None
