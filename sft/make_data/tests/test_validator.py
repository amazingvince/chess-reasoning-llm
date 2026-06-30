"""Tests for validation/validator.py — all 7 validation checks (~24 cases)."""

import chess
import pytest
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from conftest import (
    CHECKMATE_FEN,
    EN_PASSANT_FEN,
    KRK_FEN,
    PROMOTION_FEN,
    STARTING_FEN,
)
from validation.validator import (
    validate_example,
    validate_fen,
    validate_legal_moves,
    validate_move_legal,
    validate_state_tracking,
    validate_template_complete,
    validate_think_move_format,
)
from chess_llm.formats.answers import (
    extract_uci_from_move_tag as package_extract_uci_from_move_tag,
)
from chess_llm.formats.answers import (
    validate_think_move_format as package_validate_think_move_format,
)
from chess_llm.sft.validation import validate_example as package_validate_example
from chess_llm.sft.validation import validate_fen as package_validate_fen

# ── validate_fen ─────────────────────────────────────────────────────


def test_validate_fen_valid():
    assert validate_fen(STARTING_FEN) is True


def test_validate_fen_invalid():
    assert validate_fen("not/a/valid/fen") is False


def test_validate_fen_empty():
    assert validate_fen("") is False


def test_validate_fen_rejects_parseable_impossible_board():
    assert validate_fen("8/8/8/8/8/8/8/8 w - - 0 1") is False


# ── validate_legal_moves ─────────────────────────────────────────────


def test_validate_legal_moves_correct():
    board = chess.Board(STARTING_FEN)
    moves = [m.uci() for m in board.legal_moves]
    assert validate_legal_moves(STARTING_FEN, moves) is True


def test_validate_legal_moves_missing_one():
    board = chess.Board(STARTING_FEN)
    moves = [m.uci() for m in board.legal_moves]
    moves.pop()
    assert validate_legal_moves(STARTING_FEN, moves) is False


def test_validate_legal_moves_extra_one():
    board = chess.Board(STARTING_FEN)
    moves = [m.uci() for m in board.legal_moves]
    moves.append("a1a8")
    assert validate_legal_moves(STARTING_FEN, moves) is False


# ── validate_move_legal ──────────────────────────────────────────────


def test_validate_move_legal_yes():
    assert validate_move_legal(STARTING_FEN, "e2e4") is True


def test_validate_move_legal_no():
    assert validate_move_legal(STARTING_FEN, "e1e3") is False


# ── validate_state_tracking ──────────────────────────────────────────


def test_validate_state_tracking_correct():
    board = chess.Board(STARTING_FEN)
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("e7e5"))
    assert validate_state_tracking(STARTING_FEN, ["e2e4", "e7e5"], board.fen()) is True


def test_validate_state_tracking_wrong_result():
    assert validate_state_tracking(STARTING_FEN, ["e2e4", "e7e5"], STARTING_FEN) is False


def test_validate_state_tracking_chess960_castling():
    fen = "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1"
    board = chess.Board(fen, chess960=True)
    move = board.parse_uci("d1c1")
    board.push(move)

    assert validate_state_tracking(fen, ["d1c1"], board.fen(), chess960=True) is True


# ── validate_template_complete ───────────────────────────────────────


def test_validate_template_complete_clean():
    assert validate_template_complete("Position: some FEN here. All good.") is True


def test_validate_template_complete_unfilled():
    assert validate_template_complete("Position: {fen} is interesting") is False


# ── validate_think_move_format ───────────────────────────────────────


def test_think_move_format_valid():
    text = "<think>The position is equal.</think>\n<move>e2e4</move>"
    assert validate_think_move_format(text) is True


def test_think_move_format_trailing_text():
    text = "<think>The position is equal.</think>\n<move>e2e4</move>\nExtra text here"
    assert validate_think_move_format(text) is False


def test_think_move_format_missing_think():
    text = "<move>e2e4</move>"
    assert validate_think_move_format(text) is False


def test_think_move_format_with_promotion():
    text = "<think>Promote to queen.</think>\n<move>a7a8q</move>"
    assert validate_think_move_format(text) is True


def test_validator_reuses_package_answer_protocol_helpers():
    from validation import validator

    assert validate_think_move_format is package_validate_think_move_format
    assert validator._extract_uci_from_move_tag is package_extract_uci_from_move_tag


def test_validator_reuses_package_row_validation_helpers():
    from validation import validator

    assert validator.validate_example is package_validate_example
    assert validator.validate_fen is package_validate_fen


# ── validate_example — task-aware semantic checks ────────────────────


def test_validate_example_legal_move_gen_correct():
    board = chess.Board(STARTING_FEN)
    move_list = sorted(m.uci() for m in board.legal_moves)
    example = {
        "task": "2.1_legal_move_gen",
        "fen": STARTING_FEN,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"FEN: {STARTING_FEN}"},
            {"role": "assistant", "content": " ".join(move_list)},
        ],
        "metadata": {},
    }
    passed, errors = validate_example(example)
    assert passed is True, errors


def test_validate_example_legal_move_gen_wrong():
    example = {
        "task": "2.1_legal_move_gen",
        "fen": STARTING_FEN,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"FEN: {STARTING_FEN}"},
            {"role": "assistant", "content": "e2e4 d2d4"},
        ],
        "metadata": {},
    }
    passed, errors = validate_example(example)
    assert passed is False
    assert any("legal move" in e.lower() for e in errors)


def test_validate_example_move_legality_correct():
    example = {
        "task": "2.3_move_legality_check",
        "fen": STARTING_FEN,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"FEN: {STARTING_FEN}\nIs e2e4 legal?"},
            {"role": "assistant", "content": "Yes, the move is legal."},
        ],
        "metadata": {"tested_move": "e2e4"},
    }
    passed, errors = validate_example(example)
    assert passed is True, errors


def test_validate_example_move_legality_wrong():
    example = {
        "task": "2.3_move_legality_check",
        "fen": STARTING_FEN,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"FEN: {STARTING_FEN}\nIs e1e3 legal?"},
            {"role": "assistant", "content": "Yes, the move is legal."},
        ],
        "metadata": {"tested_move": "e1e3"},
    }
    passed, errors = validate_example(example)
    assert passed is False
    assert any("actually" in e.lower() for e in errors)


def test_validate_example_state_tracking_correct():
    board = chess.Board(STARTING_FEN)
    board.push(chess.Move.from_uci("e2e4"))
    result_fen = board.fen()
    example = {
        "task": "1.5_state_tracking",
        "fen": STARTING_FEN,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"FEN: {STARTING_FEN}\nMoves: e2e4"},
            {"role": "assistant", "content": result_fen},
        ],
        "metadata": {"result_fen": result_fen, "moves": "e2e4"},
    }
    passed, errors = validate_example(example)
    assert passed is True, errors


def test_validate_example_tier7_valid():
    example = {
        "task": "7.1_best_move_selection",
        "fen": STARTING_FEN,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"FEN: {STARTING_FEN}"},
            {
                "role": "assistant",
                "content": "<think>Position is equal.</think>\n<move>e2e4</move>",
            },
        ],
        "metadata": {},
    }
    passed, errors = validate_example(example)
    assert passed is True, errors


def test_validate_example_tier7_illegal_move():
    example = {
        "task": "7.1_best_move_selection",
        "fen": STARTING_FEN,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"FEN: {STARTING_FEN}"},
            {
                "role": "assistant",
                "content": "<think>Move the king.</think>\n<move>e1e3</move>",
            },
        ],
        "metadata": {},
    }
    passed, errors = validate_example(example)
    assert passed is False
    assert any("not legal" in e.lower() for e in errors)


def test_validate_example_unfilled_placeholder():
    example = {
        "task": "1.1_fen_to_board",
        "fen": STARTING_FEN,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "FEN: {fen}\nShow me the board."},
            {"role": "assistant", "content": "board here"},
        ],
        "metadata": {},
    }
    passed, errors = validate_example(example)
    assert passed is False
    assert any("placeholder" in e.lower() for e in errors)
