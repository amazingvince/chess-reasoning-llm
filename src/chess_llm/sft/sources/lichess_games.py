"""Pure helpers for parsing Lichess game rows into position records."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

import chess
import chess.pgn

from chess_llm.sft.settings import (
    DEFAULT_HF_CACHE_DIR,
    DEFAULT_HF_DATASETS,
    DEFAULT_MIN_ELO_GAMES,
)
from chess_llm.sft.sources._streaming import (
    close_streaming_dataset,
    streaming_shutdown_wait_seconds,
)

logger = logging.getLogger(__name__)

DatasetLoader = Callable[..., Iterable[Mapping]]

# Minimum plies recovered from a partially corrupt PGN before the prefix is
# worth keeping as training signal.
MIN_RECOVERED_PLIES = 4


def game_phase(ply: int) -> str:
    """Classify a zero-based ply index into a broad game phase."""
    if ply <= 30:
        return "opening"
    if ply <= 70:
        return "middlegame"
    return "endgame"


def material_balance(board: chess.Board) -> int:
    """Return white-centric material balance in centipawns."""
    values = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0,
    }
    balance = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        value = values.get(piece.piece_type, 0)
        balance += value if piece.color == chess.WHITE else -value
    return balance


def extract_game_positions(game: dict) -> Iterator[dict]:
    """Replay a game row and yield one position record before each ply."""
    moves_str = game.get("moves") or game.get("pgn") or game.get("movetext") or ""
    if not moves_str:
        return

    game_id = hashlib.sha256(moves_str.encode("utf-8")).hexdigest()[:16]
    parsed = _parse_pgn_moves(moves_str)
    if parsed is not None:
        board, moves = parsed
        yield from _positions_from_moves(board, moves, game_id=game_id)
        return

    yield from _positions_from_moves(
        chess.Board(),
        _parse_token_moves(moves_str),
        game_id=game_id,
    )


def _parse_pgn_moves(text: str) -> tuple[chess.Board, list[chess.Move]] | None:
    parsed = chess.pgn.read_game(io.StringIO(text))
    if parsed is None:
        return None

    moves = list(parsed.mainline_moves())
    if parsed.errors:
        if len(moves) < MIN_RECOVERED_PLIES:
            return None
        logger.debug(
            "PGN parse recovered %d-ply prefix from game with %d errors",
            len(moves),
            len(parsed.errors),
        )
        return parsed.board(), moves

    if not moves:
        return None
    return parsed.board(), moves


def _parse_token_moves(text: str) -> list[chess.Move]:
    text = re.sub(r"\{[^}]*\}", "", text)
    text = re.sub(r"\d+\.(?:\.\.)?", " ", text)
    board = chess.Board()
    moves: list[chess.Move] = []
    for token in text.split():
        if token in ("1-0", "0-1", "1/2-1/2", "*"):
            break
        try:
            move = chess.Move.from_uci(token)
            if move not in board.legal_moves:
                move = board.parse_san(token)
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
            try:
                move = board.parse_san(token)
            except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                return []
        board.push(move)
        moves.append(move)
    return moves


def stream_games(
    min_elo: int = DEFAULT_MIN_ELO_GAMES,
    max_games: int | None = None,
    *,
    dataset_loader: DatasetLoader | None = None,
    dataset_name: str = DEFAULT_HF_DATASETS["lichess_games"],
    data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None,
) -> Iterator[dict]:
    """Yield normalized Lichess standard game rows.

    Rows are streamed from Hugging Face by default.  Tests can inject
    ``dataset_loader`` to avoid importing optional dependencies or touching
    the network.
    """
    if max_games is not None and max_games <= 0:
        return

    uses_default_loader = dataset_loader is None
    loader = dataset_loader or _default_dataset_loader
    load_kwargs = {"split": "train", "streaming": True}
    if data_files is not None:
        load_kwargs["data_files"] = data_files

    count = 0
    rows = loader(dataset_name, **load_kwargs)
    row_iter = iter(rows)
    try:
        for row in row_iter:
            if max_games is not None and count >= max_games:
                return

            white_elo = _coerce_int(row.get("WhiteElo", row.get("white_elo")))
            black_elo = _coerce_int(row.get("BlackElo", row.get("black_elo")))
            if white_elo is None or black_elo is None:
                continue
            if white_elo < min_elo or black_elo < min_elo:
                continue

            moves_raw = str(row.get("moves") or row.get("pgn") or row.get("movetext") or "")
            if not moves_raw:
                continue

            yield {
                "pgn": row.get("pgn") or moves_raw,
                "white_elo": white_elo,
                "black_elo": black_elo,
                "result": row.get("Result", row.get("result", "*")),
                "moves": moves_raw,
            }
            count += 1
    finally:
        close_streaming_dataset(
            row_iter,
            rows,
            shutdown_wait_seconds=(
                streaming_shutdown_wait_seconds() if uses_default_loader else 0.0
            ),
        )


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _default_dataset_loader(*args, **kwargs):
    os.environ.setdefault("HF_HOME", DEFAULT_HF_CACHE_DIR)
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _positions_from_moves(
    initial_board: chess.Board,
    moves: Iterable[chess.Move],
    *,
    game_id: str | None = None,
) -> Iterator[dict]:
    board = initial_board.copy()
    for ply, move in enumerate(moves):
        if move not in board.legal_moves:
            return
        yield {
            "fen": board.fen(),
            "move_played_uci": move.uci(),
            "game_phase": game_phase(ply),
            "material_balance": material_balance(board),
            "ply": ply,
            "game_id": game_id,
        }
        board.push(move)


__all__ = [
    "extract_game_positions",
    "game_phase",
    "material_balance",
    "stream_games",
]
