"""Mechanical hanging-piece claim helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

import chess


PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

HANGING_CLAIM_CASE_ORDER = (
    "true_hanging_claim",
    "defended_decoy_claim",
    "missed_hanging_claim",
    "safe_undefended_decoy_claim",
    "true_non_hanging_claim",
)


@dataclass(frozen=True)
class HangingPieceStatus:
    square: str
    color: str
    piece: str
    attacked: bool
    defended: bool
    hanging: bool
    status: str

    @property
    def description(self) -> str:
        return f"{self.color} {self.piece} on {self.square}"


@dataclass(frozen=True)
class HangingClaimCase:
    verification_claim: str
    answer: str
    corruption_kind: str
    query_square: str
    query_piece: str
    query_color: str
    attacked: bool
    defended: bool
    hanging: bool
    claimed_hanging: bool
    verdict: str
    hanging_status: str
    correction: str


@dataclass(frozen=True)
class HangingClaimVerificationLabel:
    verdict: str
    attacked: bool
    defended: bool
    hanging: bool
    correction: str


_VERDICT_RE = re.compile(r"^Verdict:\s*(correct|incorrect)\s*$", re.IGNORECASE)
_ATTACKED_RE = re.compile(r"^Attacked:\s*(yes|no)\s*$", re.IGNORECASE)
_DEFENDED_RE = re.compile(r"^Defended:\s*(yes|no)\s*$", re.IGNORECASE)
_HANGING_RE = re.compile(r"^Hanging:\s*(yes|no)\s*$", re.IGNORECASE)
_CORRECTION_RE = re.compile(r"^Correction:\s*(.*?)\s*$", re.IGNORECASE)


def hanging_piece_statuses(board: chess.Board) -> list[HangingPieceStatus]:
    """Return non-king piece statuses in a1..h8 square order."""
    statuses: list[HangingPieceStatus] = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.piece_type == chess.KING:
            continue
        attacked = board.is_attacked_by(not piece.color, square)
        defended = board.is_attacked_by(piece.color, square)
        hanging = attacked and not defended
        if hanging:
            status = "hanging"
        elif attacked and defended:
            status = "attacked_defended"
        elif defended:
            status = "safe_defended"
        else:
            status = "safe_undefended"
        statuses.append(
            HangingPieceStatus(
                square=chess.square_name(square),
                color="white" if piece.color == chess.WHITE else "black",
                piece=PIECE_NAMES[piece.piece_type],
                attacked=attacked,
                defended=defended,
                hanging=hanging,
                status=status,
            )
        )
    return statuses


def select_hanging_claim_case(
    board: chess.Board,
    case_index: int,
) -> HangingClaimCase | None:
    """Select a deterministic verification case from a board."""
    statuses = hanging_piece_statuses(board)
    if not statuses:
        return None

    cases = [_build_case(status, kind) for kind in HANGING_CLAIM_CASE_ORDER for status in statuses]
    cases = [case for case in cases if case is not None]
    if not cases:
        return None

    preferred_kind = HANGING_CLAIM_CASE_ORDER[case_index % len(HANGING_CLAIM_CASE_ORDER)]
    for case in cases:
        if case.corruption_kind == preferred_kind:
            return case
    return cases[0]


def format_hanging_claim_verification_answer(label: HangingClaimVerificationLabel) -> str:
    """Format a hanging-claim verifier label with fixed grammar."""
    return "\n".join(
        [
            f"Verdict: {label.verdict}",
            f"Attacked: {'yes' if label.attacked else 'no'}",
            f"Defended: {'yes' if label.defended else 'no'}",
            f"Hanging: {'yes' if label.hanging else 'no'}",
            f"Correction: {label.correction}",
        ]
    )


def parse_hanging_claim_verification_answer(
    answer: str,
) -> HangingClaimVerificationLabel | None:
    """Parse fixed-grammar hanging-claim verifier output."""
    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
    if len(lines) != 5:
        return None
    verdict = _VERDICT_RE.match(lines[0])
    attacked = _ATTACKED_RE.match(lines[1])
    defended = _DEFENDED_RE.match(lines[2])
    hanging = _HANGING_RE.match(lines[3])
    correction = _CORRECTION_RE.match(lines[4])
    if not (verdict and attacked and defended and hanging and correction):
        return None
    return HangingClaimVerificationLabel(
        verdict=verdict.group(1).lower(),
        attacked=attacked.group(1).lower() == "yes",
        defended=defended.group(1).lower() == "yes",
        hanging=hanging.group(1).lower() == "yes",
        correction=correction.group(1).strip(),
    )


def hanging_piece_claim_verification_score(
    prediction: str,
    gold: str,
) -> dict[str, float]:
    """Score fixed-grammar hanging-claim verifier predictions."""
    expected = parse_hanging_claim_verification_answer(gold)
    pred = parse_hanging_claim_verification_answer(prediction)
    if expected is None or pred is None:
        return {
            "primary": 0.0,
            "verdict_accuracy": 0.0,
            "attacked_accuracy": 0.0,
            "defended_accuracy": 0.0,
            "hanging_accuracy": 0.0,
            "correction_match": 0.0,
        }

    scores = {
        "verdict_accuracy": 1.0 if pred.verdict == expected.verdict else 0.0,
        "attacked_accuracy": 1.0 if pred.attacked == expected.attacked else 0.0,
        "defended_accuracy": 1.0 if pred.defended == expected.defended else 0.0,
        "hanging_accuracy": 1.0 if pred.hanging == expected.hanging else 0.0,
        "correction_match": (
            1.0
            if _normalize_correction(pred.correction)
            == _normalize_correction(expected.correction)
            else 0.0
        ),
    }
    return {"primary": sum(scores.values()) / len(scores), **scores}


def _build_case(
    status: HangingPieceStatus,
    corruption_kind: str,
) -> HangingClaimCase | None:
    if corruption_kind == "true_hanging_claim":
        if not status.hanging:
            return None
        claimed_hanging = True
    elif corruption_kind == "defended_decoy_claim":
        if status.status != "attacked_defended":
            return None
        claimed_hanging = True
    elif corruption_kind == "missed_hanging_claim":
        if not status.hanging:
            return None
        claimed_hanging = False
    elif corruption_kind == "safe_undefended_decoy_claim":
        if status.status != "safe_undefended":
            return None
        claimed_hanging = True
    elif corruption_kind == "true_non_hanging_claim":
        if status.hanging:
            return None
        claimed_hanging = False
    else:
        return None

    verdict = "correct" if claimed_hanging == status.hanging else "incorrect"
    correction = "none" if verdict == "correct" else _claim_text(status, status.hanging)
    label = HangingClaimVerificationLabel(
        verdict=verdict,
        attacked=status.attacked,
        defended=status.defended,
        hanging=status.hanging,
        correction=correction,
    )
    return HangingClaimCase(
        verification_claim=_claim_text(status, claimed_hanging),
        answer=format_hanging_claim_verification_answer(label),
        corruption_kind=corruption_kind,
        query_square=status.square,
        query_piece=status.piece,
        query_color=status.color,
        attacked=status.attacked,
        defended=status.defended,
        hanging=status.hanging,
        claimed_hanging=claimed_hanging,
        verdict=verdict,
        hanging_status=status.status,
        correction=correction,
    )


def _claim_text(status: HangingPieceStatus, hanging: bool) -> str:
    negation = "" if hanging else " not"
    return f"{status.description} is{negation} hanging."


def _normalize_correction(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(".")


__all__ = [
    "HANGING_CLAIM_CASE_ORDER",
    "HangingClaimCase",
    "HangingClaimVerificationLabel",
    "HangingPieceStatus",
    "format_hanging_claim_verification_answer",
    "hanging_piece_claim_verification_score",
    "hanging_piece_statuses",
    "parse_hanging_claim_verification_answer",
    "select_hanging_claim_case",
]
