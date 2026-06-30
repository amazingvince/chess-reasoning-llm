"""Move-legality classification helpers shared by SFT and eval code."""

from __future__ import annotations

import re
from dataclasses import dataclass
from random import Random

import chess


LEGALITY_REASON_PHRASES: dict[str, str] = {
    "legal": "legal",
    "malformed_uci": "malformed UCI",
    "empty_source": "empty source",
    "wrong_side_piece": "wrong side piece",
    "own_piece_destination": "own piece destination",
    "illegal_piece_movement_or_blocked_path": "illegal piece movement or blocked path",
    "missing_or_invalid_promotion": "missing or invalid promotion",
    "does_not_resolve_check": "does not resolve check",
    "pinned_piece_exposes_king": "pinned piece exposes king",
    "king_would_be_in_check": "king would be in check",
}

ILLEGAL_LEGALITY_REASON_LABELS: tuple[str, ...] = (
    "empty_source",
    "wrong_side_piece",
    "own_piece_destination",
    "illegal_piece_movement_or_blocked_path",
    "missing_or_invalid_promotion",
    "does_not_resolve_check",
    "pinned_piece_exposes_king",
    "king_would_be_in_check",
)

LEGALITY_REASON_LABELS: tuple[str, ...] = ("legal", *ILLEGAL_LEGALITY_REASON_LABELS)

_PROMOTION_PIECES = {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT}
_YES_RE = re.compile(r"\byes\b|\blegal\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b|\billegal\b|\bnot\s+legal\b", re.IGNORECASE)


@dataclass(frozen=True)
class MoveLegalityClassification:
    """Structured result for a candidate UCI move in a board position."""

    move_uci: str
    is_legal: bool
    reason_label: str


def legality_reason_phrase(reason_label: str) -> str:
    """Return the model-facing phrase for a canonical reason label."""
    return LEGALITY_REASON_PHRASES.get(reason_label, reason_label.replace("_", " "))


def format_legality_answer(is_legal: bool, reason_label: str) -> str:
    """Format a compact answer that teaches both legality and reason."""
    label = "legal" if is_legal else reason_label
    phrase = legality_reason_phrase(label)
    if is_legal:
        return f"Yes, legal. Reason: {phrase}."
    return f"No, illegal. Reason: {phrase}."


def classify_move_legality(
    board: chess.Board,
    move_uci: str,
) -> MoveLegalityClassification:
    """Classify whether a UCI move is legal and why it is rejected."""
    normalized_uci = str(move_uci or "").strip().lower()
    try:
        move = chess.Move.from_uci(normalized_uci)
    except (ValueError, TypeError):
        return MoveLegalityClassification(normalized_uci, False, "malformed_uci")

    if move in board.legal_moves:
        return MoveLegalityClassification(normalized_uci, True, "legal")

    piece = board.piece_at(move.from_square)
    if piece is None:
        return MoveLegalityClassification(normalized_uci, False, "empty_source")
    if piece.color != board.turn:
        return MoveLegalityClassification(normalized_uci, False, "wrong_side_piece")

    if _requires_promotion(piece, move) and move.promotion not in _PROMOTION_PIECES:
        return MoveLegalityClassification(
            normalized_uci,
            False,
            "missing_or_invalid_promotion",
        )

    target_piece = board.piece_at(move.to_square)
    if target_piece is not None and target_piece.color == board.turn:
        return MoveLegalityClassification(normalized_uci, False, "own_piece_destination")

    if not board.is_pseudo_legal(move):
        return MoveLegalityClassification(
            normalized_uci,
            False,
            "illegal_piece_movement_or_blocked_path",
        )

    if board.is_check():
        return MoveLegalityClassification(normalized_uci, False, "does_not_resolve_check")

    if board.is_pinned(board.turn, move.from_square):
        return MoveLegalityClassification(
            normalized_uci,
            False,
            "pinned_piece_exposes_king",
        )

    return MoveLegalityClassification(normalized_uci, False, "king_would_be_in_check")


def random_illegal_move_with_reason(
    board: chess.Board,
    rng: Random,
) -> tuple[str, str] | None:
    """Return a varied illegal move and its canonical reason label."""
    reasons = list(ILLEGAL_LEGALITY_REASON_LABELS)
    rng.shuffle(reasons)
    for reason in reasons:
        move_uci = _candidate_for_reason(board, reason, rng)
        if move_uci is not None:
            return move_uci, reason

    for _ in range(80):
        from_sq = rng.choice(chess.SQUARES)
        to_sq = rng.choice(chess.SQUARES)
        if from_sq == to_sq:
            continue
        uci = chess.square_name(from_sq) + chess.square_name(to_sq)
        result = classify_move_legality(board, uci)
        if not result.is_legal and result.reason_label in ILLEGAL_LEGALITY_REASON_LABELS:
            return uci, result.reason_label
    return None


def parse_legality_yes_no_answer(text: str) -> bool | None:
    """Parse a legality answer into True, False, or unknown."""
    lower = (text or "").strip().lower()
    if not lower:
        return None

    has_no = _NO_RE.search(lower) is not None
    if has_no:
        lower_without_no = _NO_RE.sub(" ", lower)
        if _YES_RE.search(lower_without_no):
            return None
        return False

    if _YES_RE.search(lower):
        return True
    return None


def parse_legality_reason_label(text: str) -> str | None:
    """Parse a canonical legality reason label from model-facing text."""
    normalized = _normalize_reason_text(_reason_segment(text))
    if not normalized:
        return None

    for label in sorted(LEGALITY_REASON_LABELS, key=len, reverse=True):
        phrase = _normalize_reason_text(legality_reason_phrase(label))
        compact_label = _normalize_reason_text(label)
        if _contains_normalized_phrase(normalized, phrase) or _contains_normalized_phrase(
            normalized,
            compact_label,
        ):
            return label
    return None


def legality_binary_accuracy(prediction: str, gold: str) -> float:
    """Score only the yes/no legality decision."""
    expected = parse_legality_yes_no_answer(gold)
    actual = parse_legality_yes_no_answer(prediction)
    if expected is None or actual is None:
        return 0.0
    return 1.0 if actual == expected else 0.0


def legality_reason_accuracy(
    prediction: str,
    gold: str,
    *,
    gold_reason_label: str | None = None,
) -> float | None:
    """Score the reason label when a gold reason is available."""
    expected = gold_reason_label or parse_legality_reason_label(gold)
    if expected not in LEGALITY_REASON_LABELS:
        return None
    actual = parse_legality_reason_label(prediction)
    return 1.0 if actual == expected else 0.0


def _requires_promotion(piece: chess.Piece, move: chess.Move) -> bool:
    if piece.piece_type != chess.PAWN:
        return False
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    if piece.color == chess.WHITE:
        return from_rank == 6 and to_rank == 7
    return from_rank == 1 and to_rank == 0


def _candidate_for_reason(
    board: chess.Board,
    reason: str,
    rng: Random,
) -> str | None:
    if reason == "empty_source":
        return _first_matching_candidate(board, _empty_source_candidates(board, rng), reason)
    if reason == "wrong_side_piece":
        return _first_matching_candidate(board, _wrong_side_candidates(board, rng), reason)
    if reason == "own_piece_destination":
        return _first_matching_candidate(board, _own_destination_candidates(board, rng), reason)
    if reason == "missing_or_invalid_promotion":
        return _first_matching_candidate(board, _missing_promotion_candidates(board, rng), reason)
    if reason == "does_not_resolve_check":
        return _first_matching_candidate(board, _pseudo_illegal_candidates(board, rng), reason)
    if reason == "pinned_piece_exposes_king":
        return _first_matching_candidate(board, _pinned_piece_candidates(board, rng), reason)
    if reason == "king_would_be_in_check":
        return _first_matching_candidate(board, _pseudo_illegal_candidates(board, rng), reason)
    if reason == "illegal_piece_movement_or_blocked_path":
        return _first_matching_candidate(board, _own_piece_destination_scan(board, rng), reason)
    return None


def _first_matching_candidate(
    board: chess.Board,
    candidates,
    reason: str,
) -> str | None:
    for uci in candidates:
        result = classify_move_legality(board, uci)
        if not result.is_legal and result.reason_label == reason:
            return uci
    return None


def _shuffled_squares(rng: Random) -> list[int]:
    squares = list(chess.SQUARES)
    rng.shuffle(squares)
    return squares


def _empty_source_candidates(board: chess.Board, rng: Random):
    for from_sq in _shuffled_squares(rng):
        if board.piece_at(from_sq) is not None:
            continue
        for to_sq in _shuffled_squares(rng):
            if from_sq != to_sq:
                yield chess.square_name(from_sq) + chess.square_name(to_sq)


def _wrong_side_candidates(board: chess.Board, rng: Random):
    for from_sq in _shuffled_squares(rng):
        piece = board.piece_at(from_sq)
        if piece is None or piece.color == board.turn:
            continue
        for to_sq in _shuffled_squares(rng):
            if from_sq != to_sq:
                yield chess.square_name(from_sq) + chess.square_name(to_sq)


def _own_destination_candidates(board: chess.Board, rng: Random):
    own_squares = [
        sq
        for sq in chess.SQUARES
        if (piece := board.piece_at(sq)) is not None and piece.color == board.turn
    ]
    rng.shuffle(own_squares)
    for from_sq in own_squares:
        for to_sq in own_squares:
            if from_sq != to_sq:
                yield chess.square_name(from_sq) + chess.square_name(to_sq)


def _own_piece_destination_scan(board: chess.Board, rng: Random):
    for from_sq in _shuffled_squares(rng):
        piece = board.piece_at(from_sq)
        if piece is None or piece.color != board.turn:
            continue
        for to_sq in _shuffled_squares(rng):
            if from_sq != to_sq:
                yield chess.square_name(from_sq) + chess.square_name(to_sq)


def _missing_promotion_candidates(board: chess.Board, rng: Random):
    for from_sq in _shuffled_squares(rng):
        piece = board.piece_at(from_sq)
        if piece is None or piece.color != board.turn or piece.piece_type != chess.PAWN:
            continue
        direction = 8 if piece.color == chess.WHITE else -8
        forward = from_sq + direction
        if chess.SQUARES[0] <= forward <= chess.SQUARES[-1]:
            yield chess.square_name(from_sq) + chess.square_name(forward)
        for offset in (direction - 1, direction + 1):
            target = from_sq + offset
            if chess.SQUARES[0] <= target <= chess.SQUARES[-1]:
                yield chess.square_name(from_sq) + chess.square_name(target)


def _pseudo_illegal_candidates(board: chess.Board, rng: Random):
    moves = [
        move.uci()
        for move in board.pseudo_legal_moves
        if move not in board.legal_moves
    ]
    rng.shuffle(moves)
    yield from moves


def _pinned_piece_candidates(board: chess.Board, rng: Random):
    pinned_sources = [
        sq
        for sq in chess.SQUARES
        if (piece := board.piece_at(sq)) is not None
        and piece.color == board.turn
        and board.is_pinned(board.turn, sq)
    ]
    rng.shuffle(pinned_sources)
    for from_sq in pinned_sources:
        for to_sq in _shuffled_squares(rng):
            if from_sq != to_sq:
                yield chess.square_name(from_sq) + chess.square_name(to_sq)


def _reason_segment(text: str) -> str:
    lowered = str(text or "")
    match = re.search(r"\breason\s*:\s*(.+)", lowered, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1)
    return lowered


def _normalize_reason_text(text: str) -> str:
    text = str(text or "").lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    return f" {phrase} " in f" {text} "


__all__ = [
    "ILLEGAL_LEGALITY_REASON_LABELS",
    "LEGALITY_REASON_LABELS",
    "LEGALITY_REASON_PHRASES",
    "MoveLegalityClassification",
    "classify_move_legality",
    "format_legality_answer",
    "legality_binary_accuracy",
    "legality_reason_accuracy",
    "legality_reason_phrase",
    "parse_legality_reason_label",
    "parse_legality_yes_no_answer",
    "random_illegal_move_with_reason",
]
