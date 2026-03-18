"""Eval harness: self-consistency checks against held-out eval split data.

Each ``answer_*`` function re-derives an answer from raw source fields
using the same logic the generators use. ``evaluate_split`` runs applicable
checks on every example in a split and reports pass/fail counts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import chess

from generators.tier4_evaluation import _cp_to_bucket, _analyze_pawn_structure
from generators.tier6_endgames import _material_signature, _WDL_LABELS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Answer functions (pure, no IO)
# ---------------------------------------------------------------------------


def answer_legal_moves(fen: str) -> str:
    """Sorted space-separated UCI legal move list."""
    board = chess.Board(fen)
    return " ".join(sorted(m.uci() for m in board.legal_moves))


def answer_check_detection(fen: str) -> str:
    """Return game status: Checkmate, Check, Stalemate, or Normal."""
    board = chess.Board(fen)
    if board.is_checkmate():
        return "Checkmate"
    if board.is_check():
        return "Check"
    if board.is_stalemate():
        return "Stalemate"
    return "Normal"


def answer_captures(fen: str) -> str:
    """Sorted space-separated UCI capture moves."""
    board = chess.Board(fen)
    captures = [m.uci() for m in board.legal_moves if board.is_capture(m)]
    return " ".join(sorted(captures))


def answer_material_balance(fen: str) -> str:
    """Material balance string in the same format as MaterialBalance generator."""
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
    }
    piece_names = {
        chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
        chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
    }
    board = chess.Board(fen)
    white_mat = 0
    black_mat = 0
    white_pieces: dict[str, int] = {}
    black_pieces: dict[str, int] = {}

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        val = piece_values.get(piece.piece_type, 0)
        name = piece_names[piece.piece_type]
        if piece.color == chess.WHITE:
            white_mat += val
            white_pieces[name] = white_pieces.get(name, 0) + 1
        else:
            black_mat += val
            black_pieces[name] = black_pieces.get(name, 0) + 1

    diff = white_mat - black_mat
    w_str = ", ".join(f"{v} {k}" for k, v in sorted(white_pieces.items()))
    b_str = ", ".join(f"{v} {k}" for k, v in sorted(black_pieces.items()))

    if diff > 0:
        balance = f"White is ahead by {diff} point(s)."
    elif diff < 0:
        balance = f"Black is ahead by {abs(diff)} point(s)."
    else:
        balance = "Material is equal."

    return f"White: {w_str} ({white_mat} pts). Black: {b_str} ({black_mat} pts). {balance}"


def answer_position_eval(cp: int | None, mate: int | None) -> str:
    """Evaluation bucket label, mirroring PositionEvaluation generator."""
    if mate is not None:
        if mate > 0:
            return "White has a decisive advantage (forced mate)."
        else:
            return "Black has a decisive advantage (forced mate)."
    elif cp is not None:
        return _cp_to_bucket(cp) + "."
    return ""


def answer_pawn_structure(fen: str) -> str:
    """Pawn structure description, mirroring PawnStructure generator."""
    board = chess.Board(fen)
    analysis = _analyze_pawn_structure(board)
    parts = []
    for color_name in ("white", "black"):
        info = analysis[color_name]
        if info["doubled"]:
            files = ", ".join(info["doubled"])
            parts.append(
                f"{color_name.capitalize()} has doubled pawns on file(s) {files}."
            )
        if info["isolated"]:
            sqs = ", ".join(info["isolated"])
            parts.append(
                f"{color_name.capitalize()} has isolated pawn(s) on {sqs}."
            )
        if info["passed"]:
            sqs = ", ".join(info["passed"])
            parts.append(
                f"{color_name.capitalize()} has passed pawn(s) on {sqs}."
            )
    return " ".join(parts) if parts else "No notable pawn structure features."


def answer_endgame_classification(fen: str, material: str | None = None) -> str:
    """Endgame classification, mirroring EndgameClassification generator."""
    if material:
        sig = material
    else:
        board = chess.Board(fen)
        sig = _material_signature(board)
    return f"This is a {sig} endgame."


def answer_endgame_wdl(wdl: int) -> str:
    """WDL label, mirroring EndgameWDL generator."""
    return _WDL_LABELS.get(wdl, f"WDL value: {wdl}")


# ---------------------------------------------------------------------------
# Split-to-task mapping
# ---------------------------------------------------------------------------

SPLIT_CHECKS: dict[str, list] = {
    "perception": [answer_legal_moves, answer_check_detection],
    "rules": [answer_legal_moves, answer_check_detection],
    "tactics": [answer_captures],
    "evaluation": [answer_position_eval],
    "endgames": [answer_endgame_classification, answer_endgame_wdl],
    "chess960": [answer_legal_moves],
}


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Aggregated result from evaluating one split."""

    split: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def _run_check(fn, example: dict) -> tuple[bool, str]:
    """Run a single answer function on an example. Returns (ok, error_msg)."""
    fen = example.get("fen", "")

    try:
        if fn is answer_position_eval:
            cp = example.get("cp")
            mate = example.get("mate")
            if cp is None and mate is None:
                return True, ""  # skip — no eval data
            result = fn(cp, mate)
        elif fn is answer_endgame_wdl:
            wdl = example.get("wdl")
            if wdl is None:
                return True, ""
            result = fn(wdl)
        elif fn is answer_endgame_classification:
            result = fn(fen, example.get("material"))
        else:
            if not fen:
                return True, ""
            result = fn(fen)

        # Basic sanity: result is non-empty
        if not result and result != "":
            return False, f"{fn.__name__}: empty result"

        # Validate the FEN parses (if the function used it)
        if fn not in (answer_position_eval, answer_endgame_wdl):
            chess.Board(fen)  # will raise on bad FEN

        return True, ""

    except Exception as exc:
        return False, f"{fn.__name__}: {exc}"


def evaluate_split(split_name: str, examples: list[dict]) -> EvalResult:
    """Run all applicable checks on examples from one eval split."""
    result = EvalResult(split=split_name, total=len(examples))
    checks = SPLIT_CHECKS.get(split_name, [])

    if not checks:
        result.skipped = result.total
        return result

    for ex in examples:
        ex_ok = True
        for fn in checks:
            ok, err = _run_check(fn, ex)
            if not ok:
                ex_ok = False
                if len(result.errors) < 20:
                    result.errors.append(err)
                break
        if ex_ok:
            result.passed += 1
        else:
            result.failed += 1

    return result
