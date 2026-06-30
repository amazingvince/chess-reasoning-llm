"""Template context helpers for SFT prompt rendering."""

from __future__ import annotations

from typing import Any, Mapping

import chess

from chess_llm.core.board import validate_fen, variant_fen_key
from chess_llm.formats.board import render_ascii_board


def raw_is_chess960(raw: Mapping[str, Any]) -> bool:
    """Infer Chess960 mode from top-level or metadata fields."""
    if bool(raw.get("is_chess960", False)):
        return True
    metadata = raw.get("metadata")
    if isinstance(metadata, Mapping):
        if bool(metadata.get("is_chess960", False)):
            return True
        if metadata.get("chess960_id") is not None:
            return True
    if raw.get("chess960_id") is not None:
        return True
    fen = raw.get("fen")
    if fen and not validate_fen(str(fen)) and validate_fen(str(fen), chess960=True):
        return True
    return False


def normalize_chess_variant_metadata(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a row copy with Chess960 markers mirrored consistently."""
    normalized = dict(raw)
    metadata_value = raw.get("metadata")
    metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
    is_960 = raw_is_chess960(raw)

    chess960_id = raw.get("chess960_id")
    if chess960_id is None:
        chess960_id = metadata.get("chess960_id")
    if chess960_id is not None:
        normalized["chess960_id"] = chess960_id
        metadata.setdefault("chess960_id", chess960_id)
        is_960 = True

    if is_960:
        normalized["is_chess960"] = True
    elif "is_chess960" in normalized:
        normalized["is_chess960"] = bool(normalized["is_chess960"])

    normalized["metadata"] = metadata
    return normalized


def raw_fen_identity_key(raw: Mapping[str, Any]) -> str:
    """Return the variant-aware identity key for a row-like mapping."""
    return variant_fen_key(
        str(raw.get("fen", "")),
        chess960=raw_is_chess960(raw),
    )


def board_from_raw(raw: Mapping[str, Any]) -> chess.Board | None:
    """Build a python-chess board from an SFT row-like mapping."""
    fen = raw.get("fen", "")
    if not fen:
        return None
    try:
        board = chess.Board(
            str(fen),
            chess960=raw_is_chess960(raw),
        )
    except (ValueError, TypeError):
        return None
    if not board.is_valid():
        return None
    return board


def build_template_context(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Augment raw example fields with common board-state prompt fields."""
    context = dict(raw)
    board = board_from_raw(context)
    if board is None:
        return context

    context.setdefault("board", render_ascii_board(board))
    context.setdefault(
        "side_to_move",
        "white" if board.turn == chess.WHITE else "black",
    )
    castling = board.castling_xfen()
    context.setdefault(
        "castling_rights",
        castling if castling and castling != "-" else "none",
    )
    context.setdefault(
        "en_passant_square",
        chess.square_name(board.ep_square) if board.ep_square is not None else "none",
    )
    context.setdefault("halfmove_clock", board.halfmove_clock)
    context.setdefault("fullmove_number", board.fullmove_number)
    return context
