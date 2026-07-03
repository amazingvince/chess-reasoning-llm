"""Slider ray-walk helpers shared by SFT generators, validation, and evals."""

from __future__ import annotations

import chess


# Direction vectors are (file_delta, rank_delta) in fixed clockwise-from-N order.
RAY_DIRECTION_VECTORS: dict[str, tuple[int, int]] = {
    "N": (0, 1),
    "NE": (1, 1),
    "E": (1, 0),
    "SE": (1, -1),
    "S": (0, -1),
    "SW": (-1, -1),
    "W": (-1, 0),
    "NW": (-1, 1),
}

SLIDER_RAY_DIRECTIONS: dict[int, tuple[str, ...]] = {
    chess.ROOK: ("N", "E", "S", "W"),
    chess.BISHOP: ("NE", "SE", "SW", "NW"),
    chess.QUEEN: ("N", "NE", "E", "SE", "S", "SW", "W", "NW"),
}

_PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def _piece_phrase(piece: chess.Piece) -> str:
    color = "white" if piece.color == chess.WHITE else "black"
    return f"{color} {_PIECE_NAMES[piece.piece_type]}"


def walk_slider_rays(board: chess.Board, square: int) -> list[dict[str, object]]:
    """Walk every ray for the slider on *square* in fixed clockwise order.

    Returns one entry per ray with the model-facing ``Ray <DIR>:`` line text,
    the ray's UCI moves, and whether a piece (own or enemy) terminated the ray
    (``blocked``); rays that run off the board end with ``edge.`` instead.
    """
    piece = board.piece_at(square)
    if piece is None or piece.piece_type not in SLIDER_RAY_DIRECTIONS:
        raise ValueError(f"No slider on {chess.square_name(square)}")

    from_name = chess.square_name(square)
    rays: list[dict[str, object]] = []
    for direction in SLIDER_RAY_DIRECTIONS[piece.piece_type]:
        file_delta, rank_delta = RAY_DIRECTION_VECTORS[direction]
        file_index = chess.square_file(square) + file_delta
        rank_index = chess.square_rank(square) + rank_delta
        segments: list[str] = []
        moves: list[str] = []
        blocked = False
        while 0 <= file_index <= 7 and 0 <= rank_index <= 7:
            target = chess.square(file_index, rank_index)
            target_name = chess.square_name(target)
            occupant = board.piece_at(target)
            if occupant is None:
                segments.append(f"{target_name} empty")
                moves.append(from_name + target_name)
            elif occupant.color != piece.color:
                segments.append(
                    f"{target_name} {_piece_phrase(occupant)}: capture (stop, included)."
                )
                moves.append(from_name + target_name)
                blocked = True
                break
            else:
                segments.append(
                    f"{target_name} {_piece_phrase(occupant)}: own piece (stop, excluded)."
                )
                blocked = True
                break
            file_index += file_delta
            rank_index += rank_delta
        if not blocked:
            segments.append("edge.")
        rays.append(
            {
                "direction": direction,
                "line": f"Ray {direction}: {'; '.join(segments)}",
                "moves": moves,
                "blocked": blocked,
            }
        )
    return rays


def ray_walk_moves(board: chess.Board, square: int) -> list[str]:
    """Sorted UCI moves reachable along the slider's rays.

    For a side-to-move slider this equals its pseudo-legal move list.
    """
    return sorted(
        move
        for ray in walk_slider_rays(board, square)
        for move in ray["moves"]
    )


def format_ray_walk_answer(board: chess.Board, square: int) -> str:
    """Format the deterministic 2.10 ray-walk answer for one slider."""
    piece = board.piece_at(square)
    rays = walk_slider_rays(board, square)
    moves = sorted(move for ray in rays for move in ray["moves"])
    lines = [f"Piece: {chess.square_name(square)} {_piece_phrase(piece)}."]
    lines.extend(str(ray["line"]) for ray in rays)
    lines.append(f"Moves from rays: {' '.join(moves) if moves else 'none'}")
    return "\n".join(lines)


__all__ = [
    "RAY_DIRECTION_VECTORS",
    "SLIDER_RAY_DIRECTIONS",
    "format_ray_walk_answer",
    "ray_walk_moves",
    "walk_slider_rays",
]
