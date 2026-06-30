"""Source 6: Syzygy Endgame Tablebases.

WDL/DTZ probing for positions with <= 7 pieces.
"""

from __future__ import annotations

from contextlib import contextmanager
from random import Random
from typing import Iterator

import chess
import chess.syzygy


@contextmanager
def open_tablebase(path: str):
    """Open Syzygy tablebases as a context manager."""
    tb = chess.syzygy.open_tablebase(path)
    try:
        yield tb
    finally:
        tb.close()


def probe(tablebase, board: chess.Board) -> dict | None:
    """Probe WDL and DTZ for a position.

    Returns ``{wdl: int, dtz: int}`` or ``None`` if the position
    is outside tablebase coverage.

    WDL: 2=win, 1=cursed win, 0=draw, -1=blessed loss, -2=loss.
    DTZ: distance to zeroing move (positive = side to move wins).
    """
    try:
        wdl = tablebase.probe_wdl(board)
        dtz = tablebase.probe_dtz(board)
        return {"wdl": wdl, "dtz": dtz}
    except (chess.syzygy.MissingTableError, KeyError, ValueError):
        return None


def best_dtz_move(tablebase, board: chess.Board) -> str | None:
    """Return the DTZ-optimal move (UCI) or None.

    Prefers wins (smallest positive DTZ = fastest) over draws over
    losses (most negative DTZ = slowest loss).

    After pushing a move, we probe from the **opponent's** perspective.
    Negating gives *our* DTZ: positive = we're winning, 0 = draw,
    negative = we're losing.
    """
    candidates: list[tuple[str, int, int]] = []

    for move in board.legal_moves:
        board.push(move)
        result = probe(tablebase, board)
        board.pop()
        if result is None:
            continue
        our_wdl = -int(result["wdl"])
        our_dtz = -result["dtz"]
        candidates.append((move.uci(), our_wdl, our_dtz))

    if not candidates:
        return None

    best_wdl = max(candidate[1] for candidate in candidates)
    best_class = [candidate for candidate in candidates if candidate[1] == best_wdl]
    if best_wdl > 0:
        # Within the same winning WDL class, choose the fastest zeroing move.
        return min(best_class, key=lambda candidate: abs(candidate[2]))[0]
    if best_wdl == 0:
        return best_class[0][0]
    # Within the same losing WDL class, choose the slowest loss.
    return min(best_class, key=lambda candidate: candidate[2])[0]


# --- Material configurations for endgame sampling ---

MATERIAL_CONFIGS = [
    # Basic mates
    "KQK", "KRK", "KBBK", "KBNK",
    # Pawn endgames
    "KPK", "KPPK", "KPPKP", "KPKP",
    # Rook endgames
    "KRKP", "KRPKR", "KRPPKRP",
    # Minor piece endgames
    "KBPK", "KNPK", "KBKN",
    # Queen endgames
    "KQKP", "KQKR",
]


def _material_signature(board: chess.Board) -> str:
    """Return a sorted material signature like 'KRK' or 'KRPKR'."""
    white_pieces = []
    black_pieces = []
    piece_chars = {
        chess.KING: "K", chess.QUEEN: "Q", chess.ROOK: "R",
        chess.BISHOP: "B", chess.KNIGHT: "N", chess.PAWN: "P",
    }
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        ch = piece_chars.get(piece.piece_type, "")
        if piece.color == chess.WHITE:
            white_pieces.append(ch)
        else:
            black_pieces.append(ch)

    # Sort: K first, then Q R B N P
    order = "KQRBNP"
    white_pieces.sort(key=lambda c: order.index(c) if c in order else 99)
    black_pieces.sort(key=lambda c: order.index(c) if c in order else 99)
    return "".join(white_pieces) + "".join(black_pieces)


def sample_endgame_positions(
    tablebase,
    material_config: str,
    n: int,
    rng: Random | None = None,
) -> Iterator[dict]:
    """Generate random endgame positions for a given material config.

    Yields ``{fen, wdl, dtz, material}`` for positions matching
    *material_config* across the WDL/DTZ spectrum.

    Uses random piece placement and filters for legal + tablebase-covered
    positions.
    """
    rng = rng or Random()

    # Parse material config into piece lists
    pieces_white, pieces_black = _parse_material(material_config)
    generated = 0
    attempts = 0
    max_attempts = n * 200  # safety valve

    while generated < n and attempts < max_attempts:
        attempts += 1
        board = _random_placement(pieces_white, pieces_black, rng)
        if board is None:
            continue

        result = probe(tablebase, board)
        if result is None:
            continue

        yield {
            "fen": board.fen(),
            "wdl": result["wdl"],
            "dtz": result["dtz"],
            "material": material_config,
        }
        generated += 1


def _parse_material(config: str) -> tuple[list[int], list[int]]:
    """Parse 'KRPKR' into ([KING, ROOK, PAWN], [KING, ROOK])."""
    char_to_piece = {
        "K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK,
        "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN,
    }
    # Split at the second K (which starts black's pieces)
    parts = config.split("K")
    # First part is white (K + rest), remaining parts form black
    white_str = "K" + parts[1] if len(parts) > 1 else "K"
    black_str = "K" + "K".join(parts[2:]) if len(parts) > 2 else "K"

    white = [char_to_piece[c] for c in white_str if c in char_to_piece]
    black = [char_to_piece[c] for c in black_str if c in char_to_piece]
    return white, black


def _random_placement(
    white_pieces: list[int],
    black_pieces: list[int],
    rng: Random,
) -> chess.Board | None:
    """Randomly place pieces on the board. Returns None if invalid."""
    board = chess.Board.empty()
    available = list(chess.SQUARES)
    rng.shuffle(available)

    idx = 0
    for pt in white_pieces:
        if idx >= len(available):
            return None
        # Pawns can't go on rank 1 or 8
        if pt == chess.PAWN:
            valid = [s for s in available[idx:] if chess.square_rank(s) not in (0, 7)]
            if not valid:
                return None
            sq = rng.choice(valid)
            available.remove(sq)
        else:
            sq = available[idx]
            idx += 1
        board.set_piece_at(sq, chess.Piece(pt, chess.WHITE))

    for pt in black_pieces:
        if idx >= len(available):
            return None
        if pt == chess.PAWN:
            valid = [s for s in available[idx:] if chess.square_rank(s) not in (0, 7)]
            if not valid:
                return None
            sq = rng.choice(valid)
            available.remove(sq)
        else:
            sq = available[idx]
            idx += 1
        board.set_piece_at(sq, chess.Piece(pt, chess.BLACK))

    # Random side to move
    board.turn = rng.choice([chess.WHITE, chess.BLACK])

    # Validate: no checks on the non-moving side
    if not board.is_valid():
        return None

    return board


__all__ = [
    "MATERIAL_CONFIGS",
    "_material_signature",
    "_parse_material",
    "best_dtz_move",
    "open_tablebase",
    "probe",
    "sample_endgame_positions",
]
