"""Source 1: Lichess Standard Chess Games.

Stream PGN games from HuggingFace, extract FEN at each ply,
filter by Elo >= MIN_ELO_GAMES. Provides realistic position distributions.
"""

from __future__ import annotations

import logging
import re
from typing import Iterator

import chess
import chess.pgn
from datasets import load_dataset

from config.settings import HF_DATASETS, MIN_ELO_GAMES

logger = logging.getLogger(__name__)


def stream_games(
    min_elo: int = MIN_ELO_GAMES,
    max_games: int | None = None,
) -> Iterator[dict]:
    """Yield game dicts from the Lichess standard chess games dataset.

    Each dict: ``{pgn, white_elo, black_elo, result, moves}``.
    Only games where both players have Elo >= *min_elo* are yielded.
    """
    ds = load_dataset(
        HF_DATASETS["lichess_games"], split="train", streaming=True
    )
    count = 0
    for row in ds:
        if max_games is not None and count >= max_games:
            return

        white_elo = row.get("WhiteElo") or row.get("white_elo") or 0
        black_elo = row.get("BlackElo") or row.get("black_elo") or 0

        # Handle string elo values
        if isinstance(white_elo, str):
            try:
                white_elo = int(white_elo)
            except ValueError:
                continue
        if isinstance(black_elo, str):
            try:
                black_elo = int(black_elo)
            except ValueError:
                continue

        if white_elo < min_elo or black_elo < min_elo:
            continue

        moves_raw = row.get("moves") or row.get("pgn") or row.get("movetext") or ""
        yield {
            "pgn": row.get("pgn") or moves_raw,
            "white_elo": white_elo,
            "black_elo": black_elo,
            "result": row.get("Result", row.get("result", "*")),
            "moves": moves_raw,
        }
        count += 1


def _game_phase(ply: int) -> str:
    """Classify ply into game phase."""
    if ply <= 30:
        return "opening"
    elif ply <= 70:
        return "middlegame"
    else:
        return "endgame"


def _material_balance(board: chess.Board) -> int:
    """Return material balance in centipawns (positive = white advantage)."""
    values = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
              chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
    balance = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        v = values.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            balance += v
        else:
            balance -= v
    return balance


def extract_positions(game: dict) -> Iterator[dict]:
    """Replay moves from a game and yield a dict per ply.

    Each dict: ``{fen, move_played_uci, game_phase, material_balance, ply}``.
    """
    moves_str = game.get("moves") or game.get("pgn") or game.get("movetext") or ""
    if not moves_str:
        return

    # Strip PGN clock/eval comments like { [%clk 0:05:00] } before tokenizing
    moves_str = re.sub(r'\{[^}]*\}', '', moves_str)

    board = chess.Board()
    # Try splitting as UCI moves first, fallback to SAN
    tokens = moves_str.split()
    # Filter out move numbers like "1." "2."
    tokens = [t for t in tokens if not t.endswith(".") and t not in ("1-0", "0-1", "1/2-1/2", "*")]

    for ply, token in enumerate(tokens):
        try:
            # Try UCI first
            move = chess.Move.from_uci(token)
            if move not in board.legal_moves:
                # Fallback to SAN
                move = board.parse_san(token)
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
            try:
                move = board.parse_san(token)
            except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                return  # Stop if we can't parse

        fen = board.fen()
        yield {
            "fen": fen,
            "move_played_uci": move.uci(),
            "game_phase": _game_phase(ply),
            "material_balance": _material_balance(board),
            "ply": ply,
        }
        board.push(move)
