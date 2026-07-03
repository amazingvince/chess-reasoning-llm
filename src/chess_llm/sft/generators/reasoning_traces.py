"""Shared reasoning trace generation for Tier 7 tasks.

All traces use ``<think>...</think>`` and ``<move>...</move>`` format.
Traces are calibrated for a 0.8B model: 20-200 tokens.
"""

from __future__ import annotations

from random import Random

import chess

_PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}


def generate_tactical_trace(
    board: chess.Board,
    best_move: str,
    themes: list[str],
    pv_line: str,
    rng: Random,
) -> str:
    """Generate a reasoning trace for a tactical position.

    Returns ``<think>...</think><move>...</move>`` string.
    """
    move = chess.Move.from_uci(best_move)
    from_sq = chess.square_name(move.from_square)
    to_sq = chess.square_name(move.to_square)
    moving_piece = board.piece_at(move.from_square)
    piece_name = _PIECE_NAMES.get(moving_piece.piece_type, "piece") if moving_piece else "piece"

    # Build reasoning based on themes
    thoughts = []

    if "fork" in themes:
        thoughts.append(f"The {piece_name} on {from_sq} can fork multiple pieces by moving to {to_sq}.")
    elif "pin" in themes:
        thoughts.append(f"There is a pin along the line. Moving the {piece_name} to {to_sq} exploits it.")
    elif "skewer" in themes:
        thoughts.append(f"A skewer is available — the {piece_name} attacks through to a piece behind.")
    elif "discoveredAttack" in themes:
        thoughts.append(f"Moving the {piece_name} from {from_sq} discovers an attack.")
    elif any("mate" in t.lower() for t in themes):
        if board.gives_check(move):
            thoughts.append(f"This is a mating pattern. The {piece_name} delivers check from {to_sq}.")
        else:
            thoughts.append(f"This is a mating pattern. The {piece_name} move to {to_sq} tightens the mating net.")
    else:
        # Generic tactical reasoning
        captured = board.piece_at(move.to_square)
        if captured:
            cap_name = _PIECE_NAMES.get(captured.piece_type, "piece")
            thoughts.append(f"The {piece_name} on {from_sq} can capture the {cap_name} on {to_sq}.")
        else:
            thoughts.append(f"The key move is {piece_name} from {from_sq} to {to_sq}.")

    # Add PV continuation if available
    pv_moves = pv_line.split()[:4] if pv_line else []
    if len(pv_moves) > 1:
        cont = " ".join(pv_moves[1:3])
        thoughts.append(f"After {best_move}, the expected continuation is {cont}.")

    think_text = " ".join(thoughts)
    return f"<think>{think_text}</think>\n<move>{best_move}</move>"


def generate_positional_trace(
    board: chess.Board,
    best_move: str,
    cp: int | None,
    rng: Random,
) -> str:
    """Generate a reasoning trace for a positional best-move decision."""
    move = chess.Move.from_uci(best_move)
    moving_piece = board.piece_at(move.from_square)
    piece_name = _PIECE_NAMES.get(moving_piece.piece_type, "piece") if moving_piece else "piece"
    from_sq = chess.square_name(move.from_square)
    to_sq = chess.square_name(move.to_square)

    thoughts = []

    # Evaluate position context
    if cp is not None:
        if abs(cp) < 50:
            thoughts.append("The position is roughly equal.")
        elif cp > 200:
            thoughts.append("White has a significant advantage.")
        elif cp < -200:
            thoughts.append("Black has a significant advantage.")
        elif cp > 0:
            thoughts.append("White has a slight edge.")
        else:
            thoughts.append("Black has a slight edge.")

    # Is it a capture?
    captured = board.piece_at(move.to_square)
    if captured:
        cap_name = _PIECE_NAMES.get(captured.piece_type, "piece")
        thoughts.append(f"The {piece_name} captures the {cap_name} on {to_sq}.")
    else:
        # Positional reasoning
        is_center = move.to_square in (chess.E4, chess.D4, chess.E5, chess.D5)
        if is_center:
            thoughts.append(f"Moving the {piece_name} to {to_sq} improves central control.")
        elif moving_piece and moving_piece.piece_type == chess.KING:
            thoughts.append(f"The king moves to {to_sq} for safety or activity.")
        else:
            thoughts.append(f"The {piece_name} improves its position by moving to {to_sq}.")

    # Check if the move gives check
    board_copy = board.copy()
    board_copy.push(move)
    if board_copy.is_check():
        thoughts.append("This move delivers check.")

    think_text = " ".join(thoughts)
    return f"<think>{think_text}</think>\n<move>{best_move}</move>"


def generate_endgame_trace(
    board: chess.Board,
    best_move: str,
    wdl: int | None,
    dtz: int | None,
    rng: Random,
) -> str:
    """Generate a reasoning trace for an endgame position."""
    move = chess.Move.from_uci(best_move)
    moving_piece = board.piece_at(move.from_square)
    piece_name = _PIECE_NAMES.get(moving_piece.piece_type, "piece") if moving_piece else "piece"
    to_sq = chess.square_name(move.to_square)

    thoughts = []

    # WDL context
    if wdl is not None:
        if wdl == 2:
            thoughts.append("This is a theoretically won position.")
        elif wdl == 0:
            thoughts.append("This endgame is a draw with best play.")
        elif wdl == -2:
            thoughts.append("This is a lost position; we must try to hold.")
        elif wdl == 1:
            thoughts.append("This is a cursed win — technically won but the 50-move rule may interfere.")
        elif wdl == -1:
            thoughts.append("This is a blessed loss — technically lost but the 50-move rule may save us.")

    # DTZ context
    if dtz is not None and dtz != 0:
        thoughts.append(f"Distance to zeroing move: {abs(dtz)}.")

    # Move description
    captured = board.piece_at(move.to_square)
    if captured:
        cap_name = _PIECE_NAMES.get(captured.piece_type, "piece")
        thoughts.append(f"Capture the {cap_name} on {to_sq} with the {piece_name}.")
    elif move.promotion:
        promo_names = {chess.QUEEN: "queen", chess.ROOK: "rook",
                       chess.BISHOP: "bishop", chess.KNIGHT: "knight"}
        promo_name = promo_names.get(move.promotion, "queen")
        thoughts.append(f"Promote to a {promo_name} on {to_sq}.")
    else:
        thoughts.append(f"Move the {piece_name} to {to_sq}.")

    think_text = " ".join(thoughts)
    return f"<think>{think_text}</think>\n<move>{best_move}</move>"


def convert_mate_annotation(
    mate_row: dict,
    rng: Random,
) -> str:
    """Convert a MATE dataset row into a reasoning trace.

    The trace explains the choice between move_a and move_b.
    """
    better = mate_row.get("better_move", "")
    move_a = mate_row.get("move_a", "")
    move_b = mate_row.get("move_b", "")
    strategy = mate_row.get("strategy", "")
    tactic = mate_row.get("tactic", "")

    thoughts = []

    if strategy:
        thoughts.append(f"Strategic consideration: {strategy}")
    if tactic:
        thoughts.append(f"Tactical motif: {tactic}")

    if better == move_a:
        thoughts.append(f"Move {move_a} is stronger than {move_b}.")
    elif better == move_b:
        thoughts.append(f"Move {move_b} is stronger than {move_a}.")
    else:
        thoughts.append(f"The better move is {better}.")

    think_text = " ".join(thoughts)
    return f"<think>{think_text}</think>\n<move>{better}</move>"
