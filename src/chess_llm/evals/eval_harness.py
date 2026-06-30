"""Eval-split oracle self-consistency checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from chess_llm.evals.benchmark import (
    _WDL_LABELS,
    _analyze_pawn_structure,
    _cp_to_bucket,
    _material_signature,
)


def answer_legal_moves(fen: str, chess960: bool = False) -> str:
    """Sorted space-separated UCI legal move list."""
    board = chess.Board(fen, chess960=chess960)
    return " ".join(sorted(move.uci() for move in board.legal_moves))


def answer_check_detection(fen: str, chess960: bool = False) -> str:
    """Return game status: Checkmate, Check, Stalemate, or Normal."""
    board = chess.Board(fen, chess960=chess960)
    if board.is_checkmate():
        return "Checkmate"
    if board.is_check():
        return "Check"
    if board.is_stalemate():
        return "Stalemate"
    return "Normal"


def answer_captures(fen: str, chess960: bool = False) -> str:
    """Sorted space-separated UCI capture moves."""
    board = chess.Board(fen, chess960=chess960)
    captures = [move.uci() for move in board.legal_moves if board.is_capture(move)]
    if not captures:
        return "No captures available."
    return " ".join(sorted(captures))


def answer_material_balance(fen: str, chess960: bool = False) -> str:
    """Material balance string matching the MaterialBalance generator."""
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
    }
    piece_names = {
        chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
        chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
    }
    board = chess.Board(fen, chess960=chess960)
    white_mat = 0
    black_mat = 0
    white_pieces: dict[str, int] = {}
    black_pieces: dict[str, int] = {}

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        value = piece_values.get(piece.piece_type, 0)
        name = piece_names[piece.piece_type]
        if piece.color == chess.WHITE:
            white_mat += value
            white_pieces[name] = white_pieces.get(name, 0) + 1
        else:
            black_mat += value
            black_pieces[name] = black_pieces.get(name, 0) + 1

    diff = white_mat - black_mat
    w_str = ", ".join(f"{count} {piece}" for piece, count in sorted(white_pieces.items()))
    b_str = ", ".join(f"{count} {piece}" for piece, count in sorted(black_pieces.items()))

    if diff > 0:
        balance = f"White is ahead by {diff} point(s)."
    elif diff < 0:
        balance = f"Black is ahead by {abs(diff)} point(s)."
    else:
        balance = "Material is equal."

    return f"White: {w_str} ({white_mat} pts). Black: {b_str} ({black_mat} pts). {balance}"


def answer_position_eval(cp: int | None, mate: int | None) -> str:
    """Evaluation bucket label from White-perspective cp/mate fields."""
    if mate is not None:
        if mate > 0:
            return "White has a decisive advantage (forced mate)."
        return "Black has a decisive advantage (forced mate)."
    if cp is not None:
        return _cp_to_bucket(cp) + "."
    return ""


def answer_pawn_structure(fen: str, chess960: bool = False) -> str:
    """Pawn structure description matching the PawnStructure generator."""
    board = chess.Board(fen, chess960=chess960)
    analysis = _analyze_pawn_structure(board)
    parts: list[str] = []
    for color_name in ("white", "black"):
        info = analysis[color_name]
        if info["doubled"]:
            files = ", ".join(info["doubled"])
            parts.append(f"{color_name.capitalize()} has doubled pawns on file(s) {files}.")
        if info["isolated"]:
            squares = ", ".join(info["isolated"])
            parts.append(f"{color_name.capitalize()} has isolated pawn(s) on {squares}.")
        if info["passed"]:
            squares = ", ".join(info["passed"])
            parts.append(f"{color_name.capitalize()} has passed pawn(s) on {squares}.")
    return " ".join(parts) if parts else "No notable pawn structure features."


def answer_endgame_classification(
    fen: str,
    material: str | None = None,
    chess960: bool = False,
) -> str:
    """Endgame classification matching the EndgameClassification generator."""
    if material:
        signature = material
    else:
        board = chess.Board(fen, chess960=chess960)
        signature = _material_signature(board)
    return f"This is a {signature} endgame."


def answer_endgame_wdl(wdl: int) -> str:
    """WDL label matching the EndgameWDL generator."""
    return _WDL_LABELS.get(wdl, f"WDL value: {wdl}")


def answer_opening_name(fen: str, name: str, eco: str, chess960: bool = False) -> str:
    """Data quality check for opening identification rows."""
    if not name or not eco:
        return ""
    chess.Board(fen, chess960=chess960)
    return f"{name} (ECO: {eco})"


def answer_best_move_exists(
    fen: str,
    best_move: str,
    chess960: bool = False,
) -> str:
    """Verify that ``best_move`` is legal in ``fen``."""
    board = chess.Board(fen, chess960=chess960)
    if best_move not in {move.uci() for move in board.legal_moves}:
        raise ValueError(f"best_move {best_move!r} is not legal in FEN")
    return best_move


def answer_mate_choice(
    fen: str,
    move_a: str,
    move_b: str,
    better_move: str,
    chess960: bool = False,
) -> str:
    """Verify both candidate moves are legal and ``better_move`` is one of them."""
    board = chess.Board(fen, chess960=chess960)
    legal_uci = {move.uci() for move in board.legal_moves}
    if move_a not in legal_uci:
        raise ValueError(f"move_a {move_a!r} is not legal in FEN")
    if move_b not in legal_uci:
        raise ValueError(f"move_b {move_b!r} is not legal in FEN")
    if better_move not in (move_a, move_b):
        raise ValueError(
            f"better_move {better_move!r} is not one of {move_a!r}, {move_b!r}"
        )
    return better_move


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


def _example_is_chess960(example: dict) -> bool:
    metadata = example.get("metadata")
    metadata_is_960 = False
    if isinstance(metadata, dict):
        metadata_is_960 = bool(
            metadata.get("is_chess960") or metadata.get("chess960_id") is not None
        )
    return bool(
        example.get("is_chess960")
        or example.get("chess960_id") is not None
        or metadata_is_960
    )


def _run_check(fn, example: dict) -> tuple[bool | str, str]:
    """Run a single answer function on one raw eval-split row."""
    fen = example.get("fen", "")
    is_960 = _example_is_chess960(example)
    requires_valid_fen = fn not in (answer_position_eval, answer_endgame_wdl)

    if requires_valid_fen:
        if not fen:
            return _SKIP, ""
        try:
            board = chess.Board(fen, chess960=is_960)
        except (ValueError, TypeError) as exc:
            return False, f"{fn.__name__}: invalid FEN: {exc}"
        if not board.is_valid():
            return False, f"{fn.__name__}: invalid FEN"

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
            result = fn(fen, example.get("material"), chess960=is_960)
        elif fn is answer_opening_name:
            name = example.get("name", "")
            eco = example.get("eco", "")
            if not name or not eco:
                return _SKIP, ""
            result = fn(fen, name, eco, chess960=is_960)
        elif fn is answer_best_move_exists:
            best_move = (
                example.get("best_move")
                or example.get("solution_first_move")
                or ""
            )
            if not best_move:
                return _SKIP, ""
            result = fn(fen, best_move, chess960=is_960)
        elif fn is answer_mate_choice:
            move_a = example.get("move_a", "")
            move_b = example.get("move_b", "")
            better = example.get("better_move", "")
            if not (move_a and move_b and better):
                return _SKIP, ""
            result = fn(fen, move_a, move_b, better, chess960=is_960)
        else:
            if fn in (
                answer_legal_moves,
                answer_check_detection,
                answer_captures,
                answer_material_balance,
                answer_pawn_structure,
            ):
                result = fn(fen, chess960=is_960)
            else:
                result = fn(fen)

        if result is None:
            return False, f"{fn.__name__}: empty result"
        return True, ""

    except Exception as exc:
        return False, f"{fn.__name__}: {exc}"


def evaluate_split(split_name: str, examples: list[dict]) -> EvalResult:
    """Run all applicable oracle checks on raw eval-split rows."""
    result = EvalResult(split=split_name, total=len(examples))
    checks = SPLIT_CHECKS.get(split_name, [])

    if not checks:
        result.skipped = result.total
        return result

    for example in examples:
        n_pass = 0
        n_fail = 0
        for fn in checks:
            status, error = _run_check(fn, example)
            if status is _SKIP:
                continue
            if status:
                n_pass += 1
            else:
                n_fail += 1
                if len(result.errors) < 20:
                    result.errors.append(error)
        if n_fail > 0:
            result.failed += 1
        elif n_pass > 0:
            result.passed += 1
        else:
            result.skipped += 1

    return result
