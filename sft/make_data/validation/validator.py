"""
Validation utilities for generated SFT examples.

Seven checks:
1. FEN validity
2. Legal move list exact match
3. Single move legality
4. State tracking (apply moves, compare FEN)
5. Template completeness (no unfilled placeholders)
6. <think>/<move> format validity
7. Aggregated example validation (task-aware semantic checks)
"""

from __future__ import annotations

import re

import chess


def validate_fen(fen: str) -> bool:
    """Return True if *fen* is parseable by python-chess."""
    try:
        chess.Board(fen)
        return True
    except (ValueError, TypeError):
        return False


def validate_legal_moves(fen: str, moves: list[str]) -> bool:
    """Return True if *moves* exactly matches the legal move set."""
    try:
        board = chess.Board(fen)
    except (ValueError, TypeError):
        return False
    expected = {m.uci() for m in board.legal_moves}
    return set(moves) == expected


def validate_move_legal(fen: str, uci_move: str) -> bool:
    """Return True if *uci_move* is legal in *fen*."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci_move)
        return move in board.legal_moves
    except (ValueError, TypeError):
        return False


def validate_state_tracking(
    start_fen: str, moves: list[str], result_fen: str
) -> bool:
    """Apply *moves* to *start_fen* and compare to *result_fen*."""
    try:
        board = chess.Board(start_fen)
        for uci in moves:
            board.push(chess.Move.from_uci(uci))
        return board.fen() == result_fen
    except (ValueError, TypeError, AssertionError):
        return False


_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")


def validate_template_complete(text: str) -> bool:
    """Return True if *text* contains no unfilled ``{placeholder}``."""
    return _PLACEHOLDER_RE.search(text) is None


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_MOVE_RE = re.compile(r"<move>([a-h][1-8][a-h][1-8][qrbn]?)</move>")
_THINK_MOVE_STRICT_RE = re.compile(
    r"^<think>.*?</think>\s*<move>[a-h][1-8][a-h][1-8][qrbn]?</move>\s*$",
    re.DOTALL,
)


def validate_think_move_format(text: str) -> bool:
    """Return True if *text* has a valid ``<think>...</think>`` block
    followed by a valid ``<move>...</move>`` block with no trailing content."""
    return _THINK_MOVE_STRICT_RE.match(text) is not None


def _extract_uci_from_move_tag(text: str) -> str | None:
    """Extract the UCI move string from a <move>...</move> tag."""
    m = _MOVE_RE.search(text)
    return m.group(1) if m else None


def validate_example(example: dict) -> tuple[bool, list[str]]:
    """Run all applicable validations on a full example dict.

    Returns (passed, list_of_error_strings).
    """
    errors: list[str] = []
    fen = example.get("fen", "")

    # 1. FEN validity
    if not validate_fen(fen):
        errors.append(f"Invalid FEN: {fen!r}")

    messages = example.get("messages", [])

    # 5. Template completeness (user + assistant messages)
    for msg in messages:
        if msg["role"] in ("user", "assistant"):
            if not validate_template_complete(msg["content"]):
                errors.append(
                    f"Unfilled placeholder in {msg['role']} message"
                )

    task = example.get("task", "")
    metadata = example.get("metadata", {})

    # 6. Think/move format (only for tier 7 tasks)
    if task.startswith("7."):
        for msg in messages:
            if msg["role"] == "assistant" and msg["content"]:
                if not validate_think_move_format(msg["content"]):
                    errors.append("Missing or invalid <think>/<move> tags")
                # Also validate the UCI move is legal
                uci = _extract_uci_from_move_tag(msg["content"])
                if uci and validate_fen(fen):
                    if not validate_move_legal(fen, uci):
                        errors.append(
                            f"UCI move {uci!r} in <move> tag is not legal in FEN"
                        )

    # --- Task-aware semantic validation ---

    # 2.1 Legal move generation: parse assistant as move list, validate
    if task == "2.1_legal_move_gen" and validate_fen(fen):
        for msg in messages:
            if msg["role"] == "assistant" and msg["content"]:
                move_list = msg["content"].strip().split()
                if not validate_legal_moves(fen, move_list):
                    errors.append(
                        "Legal move list does not match actual legal moves"
                    )

    # 2.3 Move legality check: extract the tested move, verify answer
    if task == "2.3_move_legality_check" and validate_fen(fen):
        tested_move = metadata.get("tested_move", "")
        if tested_move:
            is_legal = validate_move_legal(fen, tested_move)
            for msg in messages:
                if msg["role"] == "assistant":
                    says_legal = "yes" in msg["content"].lower()
                    if says_legal != is_legal:
                        errors.append(
                            f"Answer says {'legal' if says_legal else 'illegal'} "
                            f"but move {tested_move!r} is actually "
                            f"{'legal' if is_legal else 'illegal'}"
                        )

    # 1.5 State tracking: verify the result FEN matches applying moves
    if task == "1.5_state_tracking" and validate_fen(fen):
        result_fen = metadata.get("result_fen", "")
        moves_str = metadata.get("moves", "")
        if result_fen and moves_str:
            move_list = moves_str.split()
            if not validate_state_tracking(fen, move_list, result_fen):
                errors.append(
                    "State tracking: applying moves does not produce result FEN"
                )

    return (len(errors) == 0, errors)
