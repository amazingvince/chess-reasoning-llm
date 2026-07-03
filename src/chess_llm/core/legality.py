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

    if (
        _requires_promotion(piece, move)
        and move.promotion not in _PROMOTION_PIECES
        and board.is_pseudo_legal(
            chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
        )
    ):
        return MoveLegalityClassification(
            normalized_uci,
            False,
            "missing_or_invalid_promotion",
        )

    castling_reason = _blocked_castling_reason(board, move)
    if castling_reason is not None:
        return MoveLegalityClassification(normalized_uci, False, castling_reason)

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


_LEGAL_FILTER_TRACE_MAX_PIECES = 12
_LEGAL_FILTER_TRACE_MAX_PSEUDO_MOVES = 32
_LEGAL_FILTER_TRACE_MAX_CHARS = 1200

_TRACE_PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def _trace_piece_phrase(piece: chess.Piece) -> str:
    color = "white" if piece.color == chess.WHITE else "black"
    return f"{color} {_TRACE_PIECE_NAMES[piece.piece_type]}"


def _trace_move_text(moves: list[str]) -> str:
    return " ".join(moves) if moves else "none"


def format_legal_filter_trace_answer(board: chess.Board) -> str | None:
    """Format the 2.11 per-piece pseudo-legal -> rejected -> legal trace.

    The ``Side to move:``/``Pieces:`` header lines and the final
    ``All legal moves:`` line are byte-identical to the 2.9 grouped format.
    Returns ``None`` when the position exceeds the trace caps
    (>12 side-to-move pieces, >32 pseudo-legal moves, or >1200 characters).
    """
    side = "white" if board.turn == chess.WHITE else "black"
    piece_squares = [
        square
        for square in chess.SQUARES
        if (piece := board.piece_at(square)) is not None and piece.color == board.turn
    ]
    if len(piece_squares) > _LEGAL_FILTER_TRACE_MAX_PIECES:
        return None

    pseudo_by_square: dict[int, list[str]] = {square: [] for square in piece_squares}
    total_pseudo = 0
    for move in board.pseudo_legal_moves:
        pseudo_by_square.setdefault(move.from_square, []).append(move.uci())
        total_pseudo += 1
    if total_pseudo > _LEGAL_FILTER_TRACE_MAX_PSEUDO_MOVES:
        return None

    legal_by_square: dict[int, list[str]] = {square: [] for square in piece_squares}
    legal_moves: list[str] = []
    for move in board.legal_moves:
        legal_by_square.setdefault(move.from_square, []).append(move.uci())
        legal_moves.append(move.uci())

    pieces: list[str] = []
    piece_lines: list[str] = []
    for square in piece_squares:
        piece = board.piece_at(square)
        if piece is None:
            continue
        square_name = chess.square_name(square)
        phrase = _trace_piece_phrase(piece)
        pieces.append(f"{square_name} {phrase}")
        pseudo = sorted(pseudo_by_square.get(square, []))
        legal = sorted(legal_by_square.get(square, []))
        rejected = sorted(set(pseudo) - set(legal))
        if rejected:
            rejected_text = "; ".join(
                f"{move_uci} {classify_move_legality(board, move_uci).reason_label}"
                for move_uci in rejected
            )
        else:
            rejected_text = "none"
        piece_lines.append(
            f"{square_name} {phrase}: pseudo-legal {_trace_move_text(pseudo)} | "
            f"rejected {rejected_text} | legal {_trace_move_text(legal)}"
        )

    answer = "\n".join(
        [
            f"Side to move: {side}.",
            f"Pieces: {'; '.join(pieces) if pieces else 'none'}.",
            "Filter by piece:",
            *piece_lines,
            f"All legal moves: {_trace_move_text(sorted(legal_moves))}",
        ]
    )
    if len(answer) > _LEGAL_FILTER_TRACE_MAX_CHARS:
        return None
    return answer


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


def _blocked_castling_reason(board: chess.Board, move: chess.Move) -> str | None:
    """Return the king-safety label for castling blocked only by king safety.

    Detects castling-shaped king moves that have castling rights and a clear
    path, so the only rule they break is king safety (castling out of,
    through, or into check).
    """
    if move.promotion is not None or not board.is_castling(move):
        return None
    king_sq = move.from_square
    rank = chess.square_rank(king_sq)
    kingside = chess.square_file(move.to_square) > chess.square_file(king_sq)
    king_to = chess.square(6 if kingside else 2, rank)
    if board.chess960:
        # Chess960 castling is encoded as king-takes-own-rook.
        required_rook = move.to_square
    else:
        if move.to_square != king_to:
            return None
        required_rook = None

    backrank = chess.BB_RANK_1 if board.turn == chess.WHITE else chess.BB_RANK_8
    rook_sq = None
    for candidate in chess.SquareSet(board.clean_castling_rights() & backrank):
        if required_rook is not None and candidate != required_rook:
            continue
        if (chess.square_file(candidate) > chess.square_file(king_sq)) == kingside:
            rook_sq = candidate
            break
    if rook_sq is None:
        return None

    rook_to = chess.square(5 if kingside else 3, rank)
    path = (
        chess.SquareSet(chess.between(king_sq, king_to))
        | chess.SquareSet(chess.between(rook_sq, rook_to))
        | chess.SquareSet(chess.BB_SQUARES[king_to] | chess.BB_SQUARES[rook_to])
    )
    blockers = chess.SquareSet(board.occupied) - chess.SquareSet(
        chess.BB_SQUARES[king_sq] | chess.BB_SQUARES[rook_sq]
    )
    if blockers & path:
        return None
    return "king_would_be_in_check"


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
        for offset in (direction, direction - 1, direction + 1):
            target = from_sq + offset
            if not chess.SQUARES[0] <= target <= chess.SQUARES[-1]:
                continue
            queen_form = chess.Move(from_sq, target, promotion=chess.QUEEN)
            if not board.is_pseudo_legal(queen_form):
                continue
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
    "format_legal_filter_trace_answer",
    "format_legality_answer",
    "legality_binary_accuracy",
    "legality_reason_accuracy",
    "legality_reason_phrase",
    "parse_legality_reason_label",
    "parse_legality_yes_no_answer",
    "random_illegal_move_with_reason",
]
