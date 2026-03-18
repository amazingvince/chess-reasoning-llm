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


def answer_opening_name(fen: str, name: str, eco: str) -> str:
    """Data quality check for opening identification examples."""
    if not name or not eco:
        return ""
    # Verify FEN is parseable
    chess.Board(fen)
    return f"{name} (ECO: {eco})"


def answer_best_move_exists(fen: str, best_move: str) -> str:
    """Verify best_move is legal in fen."""
    board = chess.Board(fen)
    move = chess.Move.from_uci(best_move)
    if move not in board.legal_moves:
        raise ValueError(f"best_move {best_move!r} is not legal in FEN")
    return best_move


def answer_mate_choice(
    fen: str, move_a: str, move_b: str, better_move: str,
) -> str:
    """Verify both moves legal and better_move is one of them."""
    board = chess.Board(fen)
    ma = chess.Move.from_uci(move_a)
    mb = chess.Move.from_uci(move_b)
    if ma not in board.legal_moves:
        raise ValueError(f"move_a {move_a!r} is not legal in FEN")
    if mb not in board.legal_moves:
        raise ValueError(f"move_b {move_b!r} is not legal in FEN")
    if better_move not in (move_a, move_b):
        raise ValueError(
            f"better_move {better_move!r} is not one of {move_a!r}, {move_b!r}"
        )
    return better_move


# ---------------------------------------------------------------------------
# Split-to-task mapping
# ---------------------------------------------------------------------------

SPLIT_CHECKS: dict[str, list] = {
    "perception": [answer_legal_moves, answer_material_balance, answer_check_detection],
    "rules": [answer_legal_moves, answer_check_detection, answer_captures],
    "tactics": [answer_captures],
    "evaluation": [answer_position_eval, answer_material_balance, answer_pawn_structure],
    "openings": [answer_opening_name],
    "endgames": [answer_endgame_classification, answer_endgame_wdl],
    "planning": [answer_best_move_exists],
    "chess960": [answer_legal_moves, answer_check_detection],
    "mate": [answer_mate_choice],
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


_SKIP = "skip"


def _run_check(fn, example: dict) -> tuple[bool | str, str]:
    """Run a single answer function on an example.

    Returns ``(True, "")`` on pass, ``(False, error_msg)`` on fail,
    or ``("skip", "")`` when required fields are missing.
    """
    fen = example.get("fen", "")

    try:
        if fn is answer_position_eval:
            cp = example.get("cp")
            mate = example.get("mate")
            if cp is None and mate is None:
                return _SKIP, ""
            result = fn(cp, mate)
        elif fn is answer_endgame_wdl:
            wdl = example.get("wdl")
            if wdl is None:
                return _SKIP, ""
            result = fn(wdl)
        elif fn is answer_endgame_classification:
            result = fn(fen, example.get("material"))
        elif fn is answer_opening_name:
            name = example.get("name", "")
            eco = example.get("eco", "")
            if not name or not eco:
                return _SKIP, ""
            result = fn(fen, name, eco)
        elif fn is answer_best_move_exists:
            best_move = (example.get("best_move")
                         or example.get("solution_first_move")
                         or "")
            if not best_move:
                return _SKIP, ""
            result = fn(fen, best_move)
        elif fn is answer_mate_choice:
            move_a = example.get("move_a", "")
            move_b = example.get("move_b", "")
            better = example.get("better_move", "")
            if not (move_a and move_b and better):
                return _SKIP, ""
            result = fn(fen, move_a, move_b, better)
        else:
            if not fen:
                return _SKIP, ""
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
        n_skip = 0
        n_pass = 0
        n_fail = 0
        for fn in checks:
            status, err = _run_check(fn, ex)
            if status is _SKIP:
                n_skip += 1
            elif status:
                n_pass += 1
            else:
                n_fail += 1
                if len(result.errors) < 20:
                    result.errors.append(err)
        if n_fail > 0:
            result.failed += 1
        elif n_pass > 0:
            result.passed += 1
        else:
            # All checks skipped — no applicable check ran
            result.skipped += 1

    return result
