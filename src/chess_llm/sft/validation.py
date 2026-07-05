"""Validation helpers for legacy-compatible SFT training rows."""

from __future__ import annotations

import re
from collections.abc import Mapping

import chess

from chess_llm.core.legality import (
    LEGALITY_REASON_LABELS,
    classify_move_legality,
    format_legal_filter_trace_answer,
    parse_legality_reason_label,
    parse_legality_yes_no_answer,
)
from chess_llm.core.rays import SLIDER_RAY_DIRECTIONS, format_ray_walk_answer
from chess_llm.core.board import is_legal_move, validate_fen as _validate_fen
from chess_llm.formats import render_ascii_board
from chess_llm.formats.answers import (
    extract_uci_from_move_tag as _extract_uci_from_move_tag,
)
from chess_llm.formats.answers import validate_think_move_format
from chess_llm.sft.best_line_trace import (
    normalize_pv_moves,
    parse_best_line_trace_answer,
    validate_pv_moves,
)
from chess_llm.sft.context import raw_is_chess960
from chess_llm.sft.step_verification import (
    format_step_verification_answer,
    label_from_metadata,
    parse_step_verification_answer,
)


_PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}
_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}
_PIECE_ORDER = (
    chess.KING,
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
    chess.PAWN,
)


def validate_fen(fen: str, chess960: bool = False) -> bool:
    """Return True if *fen* is parseable and board-valid."""
    return _validate_fen(fen, chess960=chess960)


def validate_legal_moves(
    fen: str,
    moves: list[str],
    chess960: bool = False,
) -> bool:
    """Return True if *moves* exactly matches the legal move set."""
    if not validate_fen(fen, chess960=chess960):
        return False
    try:
        board = chess.Board(fen, chess960=chess960)
    except (ValueError, TypeError):
        return False
    expected = {move.uci() for move in board.legal_moves}
    return len(moves) == len(expected) and set(moves) == expected


def _extract_legal_move_answer_moves(content: str) -> list[str]:
    """Return the final legal-move list from bare or grouped answers."""
    for line in reversed(str(content or "").splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("all legal moves:") or lowered.startswith("legal moves:"):
            moves_text = stripped.split(":", 1)[1].strip()
            if _is_no_move_answer(moves_text):
                return []
            return moves_text.split()
        # Only the final non-empty line may carry the labelled move list;
        # never scan upward past unlabelled trailing lines to a stale list.
        break
    return str(content or "").strip().split()


_NO_MOVE_ANSWER_RE = re.compile(
    r"^\s*(?:none|no\s+legal\s+moves?(?:\s+available)?\.?|no\s+moves?\.?)\s*$",
    re.IGNORECASE,
)


def _is_no_move_answer(text: str) -> bool:
    return _NO_MOVE_ANSWER_RE.match(str(text or "")) is not None


def _parse_uci_list_or_no_moves(text: str) -> list[str]:
    stripped = str(text or "").strip()
    if _is_no_move_answer(stripped):
        return []
    return stripped.split()


def validate_move_legal(
    fen: str,
    uci_move: str,
    chess960: bool = False,
) -> bool:
    """Return True if *uci_move* is legal in *fen*."""
    return is_legal_move(fen, uci_move, chess960=chess960)


def validate_state_tracking(
    start_fen: str,
    moves: list[str],
    result_fen: str,
    chess960: bool = False,
) -> bool:
    """Apply *moves* to *start_fen* and compare to *result_fen*."""
    if not validate_fen(start_fen, chess960=chess960):
        return False
    try:
        board = chess.Board(start_fen, chess960=chess960)
        for uci in moves:
            board.push(board.parse_uci(uci))
        return board.fen() == result_fen
    except (ValueError, TypeError, AssertionError):
        return False


_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")
_CANDIDATE_RATING_RE = re.compile(
    r"^\s*Candidate\s+"
    r"(?P<uci>[a-h][1-8][a-h][1-8][qrbn]?)"
    r":\s*(?P<score>[+-]?\d+cp|M-?\d+)"
    r";\s*Bucket:\s*(?P<bucket>[a-z_ ]+)\s*$",
    re.IGNORECASE,
)
_CANDIDATE_BEST_RE = re.compile(
    r"^\s*Best:\s*(?P<uci>[a-h][1-8][a-h][1-8][qrbn]?)\s*$",
    re.IGNORECASE,
)


def validate_template_complete(text: str) -> bool:
    """Return True if *text* contains no unfilled ``{placeholder}``."""
    return _PLACEHOLDER_RE.search(text) is None


_YES_RE = re.compile(r"\byes\b|\blegal\b")
_NO_RE = re.compile(r"\bno\b|\billegal\b|\bnot\s+legal\b")


def _parse_yes_no_answer(text: str) -> bool | None:
    """Parse move-legality answers into True, False, or unknown."""
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


def _assistant_contents(messages: list[dict]) -> list[str]:
    return [
        str(msg.get("content", ""))
        for msg in messages
        if msg.get("role") == "assistant"
    ]


def _state_tracking_answer_matches_result(content: str, result_fen: str) -> bool:
    """Return True when the answer is the final FEN or ends with a labelled final FEN."""
    expected = str(result_fen).strip()
    stripped = content.strip()
    if stripped == expected:
        return True

    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            return False
        label, value = line.split(":", 1)
        if label.strip().lower() in {"result fen", "final fen"}:
            return value.strip() == expected
        return False
    return False


def _piece_phrase(piece: chess.Piece | None) -> str:
    if piece is None:
        return "empty"
    color = "white" if piece.color == chess.WHITE else "black"
    return f"{color} {_PIECE_NAMES[piece.piece_type]}"


_PIECE_TYPES_BY_NAME = {name: piece_type for piece_type, name in _PIECE_NAMES.items()}


def _piece_identification_answer(
    board: chess.Board,
    metadata: Mapping,
) -> str | None:
    """Derive the 1.3 answer from the FEN and query metadata."""
    query_kind = str(metadata.get("query_kind", "") or "")
    square_name = str(metadata.get("square", "") or "")
    if query_kind == "square_piece" or (not query_kind and square_name):
        try:
            square = chess.parse_square(square_name)
        except ValueError:
            return None
        return _piece_phrase(board.piece_at(square))
    if query_kind == "locate_pieces" or (not query_kind and metadata.get("piece")):
        color_name = str(metadata.get("color", "") or "")
        piece_name = str(metadata.get("piece", "") or "")
        if color_name not in {"white", "black"}:
            return None
        piece_type = _PIECE_TYPES_BY_NAME.get(piece_name)
        if piece_type is None:
            return None
        color = chess.WHITE if color_name == "white" else chess.BLACK
        squares = sorted(
            chess.square_name(square)
            for square in chess.SQUARES
            if (piece := board.piece_at(square)) is not None
            and piece.color == color
            and piece.piece_type == piece_type
        )
        return " ".join(squares)
    return None


def _side_piece_inventory(board: chess.Board) -> list[dict[str, str]]:
    inventory = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.color != board.turn:
            continue
        inventory.append(
            {
                "square": chess.square_name(square),
                "piece": _piece_phrase(piece),
            }
        )
    return inventory


def _side_piece_inventory_answer(board: chess.Board) -> tuple[str, list[dict[str, str]]]:
    side = "white" if board.turn == chess.WHITE else "black"
    inventory = _side_piece_inventory(board)
    pieces = "; ".join(
        f"{item['square']} {item['piece']}"
        for item in inventory
    )
    if not pieces:
        pieces = "none"
    return f"Side to move: {side}.\nPieces: {pieces}.", inventory


def _material_summary(board: chess.Board) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for color_name, color in (("white", chess.WHITE), ("black", chess.BLACK)):
        inventory: dict[str, list[str]] = {name: [] for name in _PIECE_NAMES.values()}
        counts: dict[str, int] = {name: 0 for name in _PIECE_NAMES.values()}
        values: dict[str, int] = {name: 0 for name in _PIECE_NAMES.values()}
        total = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.color != color:
                continue
            name = _PIECE_NAMES[piece.piece_type]
            value = _PIECE_VALUES[piece.piece_type]
            inventory[name].append(chess.square_name(square))
            counts[name] += 1
            values[name] += value
            total += value
        summary[color_name] = {
            "inventory": inventory,
            "counts": counts,
            "values": values,
            "total": total,
        }
    return summary


def _material_inventory_vector(summary: dict[str, object]) -> str:
    inventory = summary["inventory"]
    assert isinstance(inventory, dict)
    parts = []
    for piece_type in _PIECE_ORDER:
        name = _PIECE_NAMES[piece_type]
        squares = inventory.get(name, [])
        square_text = ",".join(squares) if squares else "none"
        parts.append(f"{name}={square_text}")
    return "; ".join(parts)


def _material_count_vector_text(summary: dict[str, object]) -> str:
    counts = summary["counts"]
    assert isinstance(counts, dict)
    return "; ".join(
        f"{_PIECE_NAMES[piece_type]}={counts.get(_PIECE_NAMES[piece_type], 0)}"
        for piece_type in _PIECE_ORDER
    )


def _material_value_vector_text(summary: dict[str, object]) -> str:
    values = summary["values"]
    assert isinstance(values, dict)
    parts = [
        f"{_PIECE_NAMES[piece_type]}={values.get(_PIECE_NAMES[piece_type], 0)}"
        for piece_type in _PIECE_ORDER
    ]
    parts.append(f"total={int(summary['total'])}")
    return "; ".join(parts)


def _material_balance_sentence(white_total: int, black_total: int) -> str:
    balance = white_total - black_total
    if balance > 0:
        return f"White is up {balance} point(s) of material."
    if balance < 0:
        return f"Black is up {abs(balance)} point(s) of material."
    return "Material is equal."


def _material_decomposition_answer(task: str, board: chess.Board) -> str:
    summary = _material_summary(board)
    white = summary["white"]
    black = summary["black"]
    if task == "1.15_material_inventory":
        return (
            f"White inventory: {_material_inventory_vector(white)}.\n"
            f"Black inventory: {_material_inventory_vector(black)}."
        )
    if task == "1.16_material_piece_counts":
        return (
            f"White counts: {_material_count_vector_text(white)}.\n"
            f"Black counts: {_material_count_vector_text(black)}."
        )
    if task == "1.17_material_value_totals":
        return (
            f"White values: {_material_value_vector_text(white)}.\n"
            f"Black values: {_material_value_vector_text(black)}."
        )
    white_total = int(white["total"])
    black_total = int(black["total"])
    return "\n".join(
        [
            (
                "Inventory: "
                f"white {_material_inventory_vector(white)} | "
                f"black {_material_inventory_vector(black)}"
            ),
            (
                "Counts: "
                f"white {_material_count_vector_text(white)} | "
                f"black {_material_count_vector_text(black)}"
            ),
            f"Values: white total={white_total}; black total={black_total}",
            f"Balance: {_material_balance_sentence(white_total, black_total)}",
        ]
    )


def _move_text(moves: list[str], *, empty: str = "none") -> str:
    return " ".join(moves) if moves else empty


def _pseudo_legal_moves_from_square(board: chess.Board, square: int) -> list[str]:
    return sorted(
        move.uci()
        for move in board.pseudo_legal_moves
        if move.from_square == square
    )


def _legal_moves_from_square(board: chess.Board, square: int) -> list[str]:
    return sorted(
        move.uci()
        for move in board.legal_moves
        if move.from_square == square
    )


def _rejected_move_text(board: chess.Board, moves: list[str]) -> str:
    if not moves:
        return "none"
    return "; ".join(
        f"{move_uci} {classify_move_legality(board, move_uci).reason_label}"
        for move_uci in moves
    )


def _legal_moves_by_piece(board: chess.Board) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None and piece.color == board.turn:
            grouped[chess.square_name(square)] = []
    for move in sorted(board.legal_moves, key=lambda item: item.uci()):
        grouped.setdefault(chess.square_name(move.from_square), []).append(move.uci())
    return grouped


def _legal_moves_by_piece_answer(board: chess.Board) -> str:
    side = "white" if board.turn == chess.WHITE else "black"
    grouped = _legal_moves_by_piece(board)
    pieces = []
    move_lines = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.color != board.turn:
            continue
        square_name = chess.square_name(square)
        phrase = _piece_phrase(piece)
        pieces.append(f"{square_name} {phrase}")
        moves = grouped.get(square_name, [])
        move_lines.append(f"{square_name} {phrase}: {_move_text(moves, empty='no legal moves')}")
    legal_moves = sorted(move.uci() for move in board.legal_moves)
    return (
        f"Side to move: {side}.\n"
        f"Pieces: {'; '.join(pieces) if pieces else 'none'}.\n"
        "Moves by piece:\n"
        + "\n".join(move_lines)
        + f"\nAll legal moves: {_move_text(legal_moves)}"
    )


def _king_safety_answer(board: chess.Board, move_uci: str) -> str:
    classification = classify_move_legality(board, move_uci)
    try:
        move = chess.Move.from_uci(move_uci)
        pseudo = board.is_pseudo_legal(move)
    except (TypeError, ValueError):
        pseudo = False
    return "\n".join(
        [
            f"Move: {move_uci}.",
            f"Pseudo-legal: {'yes' if pseudo else 'no'}.",
            f"King safe after move: {'yes' if classification.is_legal else 'no'}.",
            (
                "Final: legal; legal."
                if classification.is_legal
                else f"Final: illegal; {classification.reason_label}."
            ),
        ]
    )


def _legal_decomposition_answer(
    task: str,
    board: chess.Board,
    metadata: Mapping,
) -> str | None:
    if task in {"2.6_piece_pseudo_legal_moves", "2.7_piece_legal_filter"}:
        source_square = metadata.get("source_square") or metadata.get("square")
        if not source_square:
            return None
        try:
            square = chess.parse_square(str(source_square))
        except ValueError:
            return None
        pseudo = _pseudo_legal_moves_from_square(board, square)
        if task == "2.6_piece_pseudo_legal_moves":
            return f"Pseudo-legal moves from {source_square}: {_move_text(pseudo)}"
        legal = _legal_moves_from_square(board, square)
        rejected = sorted(set(pseudo) - set(legal))
        return "\n".join(
            [
                f"Pseudo-legal from {source_square}: {_move_text(pseudo)}.",
                f"Legal: {_move_text(legal)}.",
                f"Rejected: {_rejected_move_text(board, rejected)}.",
            ]
        )
    if task == "2.8_king_safety_filter":
        tested_move = metadata.get("tested_move") or metadata.get("move")
        if not tested_move:
            return None
        return _king_safety_answer(board, str(tested_move))
    if task == "2.9_legal_moves_by_piece":
        return _legal_moves_by_piece_answer(board)
    return None


def _piece_fen_char(piece: chess.Piece | None) -> str:
    if piece is None:
        return "1"
    return piece.symbol()


def _fen_rank_row(board: chess.Board, rank: int) -> str:
    if rank < 1 or rank > 8:
        raise ValueError(f"rank must be in 1..8, got {rank!r}")
    return board.board_fen().split("/")[8 - rank]


def _square_coordinates_answer(square_name: str) -> str:
    square = chess.parse_square(square_name)
    file_name = chess.FILE_NAMES[chess.square_file(square)]
    rank_name = str(chess.square_rank(square) + 1)
    fen_row_from_top = 8 - chess.square_rank(square)
    file_index = chess.square_file(square) + 1
    return (
        f"{square_name}: file={file_name}; rank={rank_name}; "
        f"fen_row_from_top={fen_row_from_top}; file_index={file_index}"
    )


def _expand_fen_rank_row(row: str) -> list[str]:
    cells: list[str] = []
    for char in row:
        if char.isdigit():
            cells.extend(["1"] * int(char))
        else:
            cells.append(char)
    if len(cells) != 8:
        raise ValueError(f"FEN rank row must expand to 8 cells, got {row!r}")
    return cells


def _compress_fen_rank_cells(cells: list[str]) -> str:
    if len(cells) != 8:
        raise ValueError(f"FEN rank cells must contain 8 cells, got {len(cells)}")
    parts: list[str] = []
    empty_count = 0
    for cell in cells:
        if cell == "1":
            empty_count += 1
            continue
        if empty_count:
            parts.append(str(empty_count))
            empty_count = 0
        parts.append(cell)
    if empty_count:
        parts.append(str(empty_count))
    return "".join(parts)


def _fen_rank_expansion_answer(rank: int, row: str) -> str:
    cells = _expand_fen_rank_row(row)
    assignments = [
        f"{file_name}={cell}"
        for file_name, cell in zip(chess.FILE_NAMES, cells, strict=True)
    ]
    return f"rank {rank}: {'; '.join(assignments)}"


def _rank_cell_edit_answer(before_row: str, rank: int, file_name: str, after_fen: str) -> str:
    cells = _expand_fen_rank_row(before_row)
    file_index = chess.FILE_NAMES.index(file_name)
    cells[file_index] = after_fen
    after_row = _compress_fen_rank_cells(cells)
    return f"rank {rank}: {before_row} -> {after_row}"


def _board_edit_answer(board_fen_after: str) -> str:
    return f"Result board FEN: {board_fen_after}"


def _changed_square_edits(
    before: chess.Board,
    after: chess.Board,
    preferred_order: list[int] | None = None,
) -> list[dict[str, str]]:
    changed = []
    for square in chess.SQUARES:
        before_piece = before.piece_at(square)
        after_piece = after.piece_at(square)
        if before_piece == after_piece:
            continue
        changed.append(
            {
                "square": chess.square_name(square),
                "before": _piece_phrase(before_piece),
                "after": _piece_phrase(after_piece),
                "before_fen": _piece_fen_char(before_piece),
                "after_fen": _piece_fen_char(after_piece),
            }
        )
    if preferred_order:
        rank = {square: index for index, square in enumerate(preferred_order)}
        changed.sort(
            key=lambda item: (
                rank.get(chess.parse_square(item["square"]), len(rank)),
                item["square"],
            )
        )
    return changed


def _move_square_edits_answer(
    fen: str,
    move_uci: str,
    *,
    chess960: bool = False,
) -> str | None:
    try:
        before = chess.Board(fen, chess960=chess960)
        move = before.parse_uci(move_uci)
        after = before.copy(stack=False)
        after.push(move)
    except (ValueError, TypeError, AssertionError):
        return None

    changed_squares = _changed_square_edits(
        before,
        after,
        preferred_order=[move.from_square, move.to_square],
    )
    lookups = [
        f"{item['square']}={item['before']}"
        for item in changed_squares
    ]
    square_edits = [
        f"{item['square']} {item['before']}->{item['after']}"
        for item in changed_squares
    ]
    rank_edits = []
    for item in changed_squares:
        square = chess.parse_square(item["square"])
        file_name = chess.FILE_NAMES[chess.square_file(square)]
        rank_name = str(chess.square_rank(square) + 1)
        rank_edits.append(
            f"rank {rank_name} {file_name} {item['before_fen']}->{item['after_fen']}"
        )
    return "\n".join(
        [
            f"Lookup: {'; '.join(lookups)}.",
            f"Squares: {'; '.join(square_edits)}.",
            f"Ranks: {'; '.join(rank_edits)}.",
        ]
    )


def _captured_piece_for_move(
    board: chess.Board,
    move: chess.Move,
) -> tuple[chess.Piece | None, str]:
    if not board.is_capture(move):
        return None, ""
    if board.is_en_passant(move):
        offset = -8 if board.turn == chess.WHITE else 8
        square = move.to_square + offset
    else:
        square = move.to_square
    return board.piece_at(square), chess.square_name(square)


def _move_trace_description(board: chess.Board, move: chess.Move) -> str:
    piece = board.piece_at(move.from_square)
    if board.is_castling(move):
        side = "kingside" if chess.square_file(move.to_square) > chess.square_file(move.from_square) else "queenside"
        sentence = f"Move 1: {_piece_phrase(piece)} {move.uci()} ({side} castling)."
    else:
        sentence = f"Move 1: {_piece_phrase(piece)} {move.uci()}."

    captured_piece, captured_square = _captured_piece_for_move(board, move)
    if captured_piece is not None:
        sentence += f" Captures {_piece_phrase(captured_piece)} on {captured_square}."
    if move.promotion:
        promoted_piece = chess.Piece(move.promotion, piece.color if piece else board.turn)
        sentence += f" Promotes to {_piece_phrase(promoted_piece)}."
    return sentence


def _fen_assembly_answer(
    fen: str,
    move_uci: str,
    *,
    chess960: bool = False,
) -> tuple[str, str] | None:
    try:
        before = chess.Board(fen, chess960=chess960)
        move = before.parse_uci(move_uci)
        local_edits = _move_square_edits_answer(fen, move_uci, chess960=chess960)
        if local_edits is None:
            return None
        after = before.copy(stack=False)
        after.push(move)
    except (ValueError, TypeError, AssertionError):
        return None

    result_fen = after.fen()
    return (
        "\n".join(
            [
                _move_trace_description(before, move),
                local_edits,
                f"Result FEN: {result_fen}",
            ]
        ),
        result_fen,
    )


def _fen_row_application_answer(
    fen: str,
    move_uci: str,
    *,
    chess960: bool = False,
) -> tuple[str, str] | None:
    """Derive the 1.10 answer (rank-row rewrites + result FEN) from the FEN."""
    try:
        before = chess.Board(fen, chess960=chess960)
        move = before.parse_uci(move_uci)
        after = before.copy(stack=False)
        after.push(move)
    except (ValueError, TypeError, AssertionError):
        return None

    changed_squares = _changed_square_edits(
        before,
        after,
        preferred_order=[move.from_square, move.to_square],
    )
    before_rows = before.board_fen().split("/")
    after_rows = after.board_fen().split("/")
    affected_ranks: list[int] = []
    for item in changed_squares:
        rank = chess.square_rank(chess.parse_square(item["square"]))
        if rank not in affected_ranks:
            affected_ranks.append(rank)
    rewrites = [
        f"rank {rank + 1} {before_rows[7 - rank]}->{after_rows[7 - rank]}"
        for rank in affected_ranks
    ]
    result_fen = after.fen()
    answer = "\n".join(
        [
            f"Rows: {'; '.join(rewrites)}.",
            f"Result FEN: {result_fen}",
        ]
    )
    return answer, result_fen


def _expected_target_move(metadata: dict) -> str | None:
    for key in (
        "target_move",
        "target_move_uci",
        "best_move",
        "solution_first_move",
        "better_move",
    ):
        move = metadata.get(key)
        if isinstance(move, str) and move.strip():
            return move.strip().lower()
    return None


def _require_exact_assistant_answer(
    messages: list[dict],
    expected: str,
    error_label: str,
) -> list[str]:
    errors: list[str] = []
    contents = _assistant_contents(messages)
    if not contents:
        return [f"{error_label}: missing assistant answer"]
    for content in contents:
        if content.strip() != expected.strip():
            errors.append(f"{error_label}: assistant answer does not match expected answer")
    return errors


def _validate_candidate_ratings_answer(
    messages: list[dict],
    fen: str,
    metadata: dict,
    *,
    chess960: bool = False,
) -> list[str]:
    errors: list[str] = []
    contents = _assistant_contents(messages)
    if not contents:
        return ["Candidate ratings: missing assistant answer"]

    expected_moves = _candidate_rating_metadata_moves(metadata)
    expected_best = _expected_target_move(metadata)
    legal_moves: set[str] | None = None
    if validate_fen(fen, chess960=chess960):
        try:
            board = chess.Board(fen, chess960=chess960)
            legal_moves = {move.uci() for move in board.legal_moves}
        except (ValueError, TypeError):
            legal_moves = None

    for content in contents:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) != 6:
            errors.append("Candidate ratings answer must contain exactly 6 non-empty lines")
            continue
        candidate_moves: list[str] = []
        for line in lines[:5]:
            match = _CANDIDATE_RATING_RE.match(line)
            if match is None:
                errors.append("Candidate ratings line does not match fixed grammar")
                continue
            uci = match.group("uci").lower()
            candidate_moves.append(uci)
            if legal_moves is not None and uci not in legal_moves:
                errors.append(f"Candidate ratings move {uci!r} is not legal in FEN")
        best_match = _CANDIDATE_BEST_RE.match(lines[5])
        if best_match is None:
            errors.append("Candidate ratings final line must be Best: <uci>")
            best_move = None
        else:
            best_move = best_match.group("uci").lower()
            if legal_moves is not None and best_move not in legal_moves:
                errors.append(f"Candidate ratings best move {best_move!r} is not legal in FEN")

        if len(set(candidate_moves)) != len(candidate_moves):
            errors.append("Candidate ratings moves must be unique")
        if expected_moves and candidate_moves != expected_moves:
            errors.append("Candidate ratings moves do not match metadata candidate_ratings")
        if expected_best and best_move != expected_best:
            errors.append(
                f"Candidate ratings best move {best_move!r} does not match "
                f"target move {expected_best!r}"
            )
        if best_move is not None and best_move not in candidate_moves:
            errors.append("Candidate ratings best move must be one of the candidates")
    return errors


def _candidate_rating_metadata_moves(metadata: dict) -> list[str]:
    ratings = metadata.get("candidate_ratings")
    if not isinstance(ratings, list):
        return []
    moves: list[str] = []
    for item in ratings[:5]:
        if not isinstance(item, Mapping):
            return []
        uci = item.get("uci") or item.get("move")
        if not isinstance(uci, str) or not uci.strip():
            return []
        moves.append(uci.strip().lower())
    return moves


def _validate_best_line_trace_answer(
    messages: list[dict],
    fen: str,
    metadata: dict,
    *,
    chess960: bool = False,
) -> list[str]:
    errors: list[str] = []
    contents = _assistant_contents(messages)
    if not contents:
        return ["Best-line trace: missing assistant answer"]

    expected_answer = metadata.get("expected_answer")
    expected_answer_text = expected_answer if isinstance(expected_answer, str) else None
    expected_move = _expected_target_move(metadata)
    expected_pv = normalize_pv_moves(metadata.get("pv") or metadata.get("pv_line"))

    for content in contents:
        parsed = parse_best_line_trace_answer(content)
        if parsed is None:
            errors.append("Best-line trace answer does not match fixed grammar")
            continue
        if expected_answer_text is not None and content.strip() != expected_answer_text.strip():
            errors.append("Best-line trace answer does not match metadata expected_answer")

        root = str(parsed["root"])
        best = str(parsed["best"])
        move = str(parsed["move"])
        pv = [str(item) for item in parsed["pv"]]
        if root != best or root != move:
            errors.append("Best-line trace root, Best line, and <move> tag must match")
        if expected_move is not None and move != expected_move:
            errors.append(
                f"Best-line trace move {move!r} does not match target move "
                f"{expected_move!r}"
            )
        if not pv or pv[0] != root:
            errors.append("Best-line trace PV must begin with the root move")
        if expected_pv and pv != expected_pv:
            errors.append("Best-line trace PV does not match metadata pv")
        if validate_fen(fen, chess960=chess960) and not validate_pv_moves(
            fen,
            pv,
            chess960=chess960,
        ):
            errors.append("Best-line trace PV is not legal from FEN")
    return errors


def _validate_step_verification_answer(
    messages: list[dict],
    metadata: dict,
) -> list[str]:
    errors: list[str] = []
    contents = _assistant_contents(messages)
    if not contents:
        return ["Step verification: missing assistant answer"]

    expected_label = label_from_metadata(metadata)
    if expected_label is None:
        return ["Step verification: missing or invalid verifier metadata"]
    expected_answer = format_step_verification_answer(expected_label)

    for content in contents:
        parsed = parse_step_verification_answer(content)
        if parsed is None:
            errors.append("Step verification answer does not match fixed grammar")
            continue
        if content.strip() != expected_answer.strip():
            errors.append("Step verification answer does not match metadata label")
    return errors


def _check_state_label(board: chess.Board) -> tuple[str, str]:
    if board.is_checkmate():
        return "checkmate", "Checkmate."
    if board.is_stalemate():
        return "stalemate", "Stalemate."
    if board.is_check():
        return "check", "Check."
    return "normal", "Normal position -- no check, checkmate, or stalemate."


def _normalize_messages(value: object) -> tuple[list[dict[str, str]], list[str]]:
    if not isinstance(value, list):
        return [], ["messages must be a list"]

    messages: list[dict[str, str]] = []
    errors: list[str] = []
    for idx, msg in enumerate(value):
        if not isinstance(msg, Mapping):
            errors.append(f"Message {idx} must be an object")
            continue

        role = msg.get("role")
        content = msg.get("content")
        valid = True
        if not isinstance(role, str) or not role:
            errors.append(f"Message {idx} missing string role")
            valid = False
        elif role not in {"system", "user", "assistant"}:
            errors.append(f"Message {idx} has unsupported role {role!r}")
            valid = False

        if not isinstance(content, str):
            errors.append(f"Message {idx} missing string content")
            valid = False

        if valid:
            messages.append({"role": role, "content": content})

    return messages, errors


def _valid_tier7_move_answer_format(task: str, content: str) -> bool:
    if task == "7.11_history_best_move":
        uci = _extract_uci_from_move_tag(content)
        return uci is not None and content == f"<move>{uci}</move>"
    return validate_think_move_format(content)


def validate_example(example: object) -> tuple[bool, list[str]]:
    """Run task-aware validation on one legacy-compatible SFT row."""
    if not isinstance(example, Mapping):
        return False, ["Example row must be a JSON object"]

    errors: list[str] = []
    task_value = example.get("task", "")
    task = task_value if isinstance(task_value, str) else ""
    if not task:
        errors.append("Missing task id")

    tier_value = example.get("tier")
    if tier_value is not None and (
        isinstance(tier_value, bool) or not isinstance(tier_value, int)
    ):
        errors.append("tier must be an integer")

    metadata_value = example.get("metadata", {})
    if metadata_value is None:
        metadata: dict = {}
    elif isinstance(metadata_value, Mapping):
        metadata = dict(metadata_value)
    else:
        metadata = {}
        errors.append("metadata must be an object")

    fen_value = example.get("fen", "")
    if isinstance(fen_value, str):
        fen = fen_value
    else:
        fen = ""
        errors.append("FEN must be a string")
    is_960 = raw_is_chess960(example)

    if not validate_fen(fen, chess960=is_960):
        errors.append(f"Invalid FEN: {fen!r}")

    messages, message_errors = _normalize_messages(example.get("messages", []))
    errors.extend(message_errors)
    for msg in messages:
        if msg["role"] in ("user", "assistant"):
            if not validate_template_complete(msg["content"]):
                errors.append(f"Unfilled placeholder in {msg['role']} message")

    if task == "7.8_candidate_ratings":
        errors.extend(
            _validate_candidate_ratings_answer(
                messages,
                fen,
                metadata,
                chess960=is_960,
            )
        )
    elif task == "7.10_best_line_trace":
        errors.extend(
            _validate_best_line_trace_answer(
                messages,
                fen,
                metadata,
                chess960=is_960,
            )
        )
    elif task == "7.9_step_verification":
        errors.extend(_validate_step_verification_answer(messages, metadata))
    elif task.startswith("7."):
        expected_move = _expected_target_move(metadata)
        for msg in messages:
            if msg["role"] == "assistant" and msg["content"]:
                if not _valid_tier7_move_answer_format(task, msg["content"]):
                    errors.append("Missing or invalid <think>/<move> tags")
                uci = _extract_uci_from_move_tag(msg["content"])
                if uci and validate_fen(fen, chess960=is_960):
                    if not validate_move_legal(fen, uci, chess960=is_960):
                        errors.append(
                            f"UCI move {uci!r} in <move> tag is not legal in FEN"
                        )
                    if expected_move is not None and uci.lower() != expected_move:
                        errors.append(
                            f"UCI move {uci!r} in <move> tag does not match "
                            f"target move {expected_move!r}"
                        )

    if task == "1.1_fen_to_board" and validate_fen(fen, chess960=is_960):
        board = chess.Board(fen, chess960=is_960)
        errors.extend(
            _require_exact_assistant_answer(
                messages,
                render_ascii_board(board),
                "Board rendering",
            )
        )

    if task == "1.2_board_to_fen" and validate_fen(fen, chess960=is_960):
        errors.extend(
            _require_exact_assistant_answer(
                messages,
                fen,
                "Board-to-FEN answer",
            )
        )

    if task == "1.3_piece_identification" and validate_fen(fen, chess960=is_960):
        board = chess.Board(fen, chess960=is_960)
        actual_answer = _piece_identification_answer(board, metadata)
        if actual_answer is None:
            errors.append("Missing or invalid metadata for piece identification")
        else:
            expected_answer = metadata.get("expected_answer")
            if expected_answer is not None and str(expected_answer) != actual_answer:
                errors.append(
                    "Piece identification expected_answer metadata does not match actual FEN"
                )
            errors.extend(
                _require_exact_assistant_answer(
                    messages,
                    actual_answer,
                    "Piece identification",
                )
            )

    if task == "1.4_piece_counting":
        expected_answer = metadata.get("expected_answer", "")
        if not expected_answer:
            errors.append("Missing expected_answer metadata for piece counting")
        else:
            errors.extend(
                _require_exact_assistant_answer(
                    messages,
                    str(expected_answer),
                    "Piece counting",
                )
            )

    if task in {
        "1.15_material_inventory",
        "1.16_material_piece_counts",
        "1.17_material_value_totals",
        "1.18_material_balance_trace",
    } and validate_fen(fen, chess960=is_960):
        board = chess.Board(fen, chess960=is_960)
        actual_answer = _material_decomposition_answer(task, board)
        expected_answer = metadata.get("expected_answer", "")
        if not expected_answer:
            errors.append("Missing expected_answer metadata for material decomposition")
        elif str(expected_answer) != actual_answer:
            errors.append("Material decomposition expected_answer metadata does not match actual FEN")
        errors.extend(
            _require_exact_assistant_answer(
                messages,
                actual_answer,
                "Material decomposition",
            )
        )

    if task == "1.6_square_lookup":
        expected_answer = metadata.get("expected_answer", "")
        square = metadata.get("square", "")
        if not expected_answer or not square:
            errors.append("Missing square/expected_answer metadata for square lookup")
        elif validate_fen(fen, chess960=is_960):
            try:
                board = chess.Board(fen, chess960=is_960)
                square_index = chess.parse_square(str(square))
                actual_answer = f"{square}={_piece_phrase(board.piece_at(square_index))}"
            except (ValueError, TypeError):
                actual_answer = ""
                errors.append("Invalid square metadata for square lookup")
            if actual_answer and str(expected_answer) != actual_answer:
                errors.append("Square lookup metadata does not match actual FEN")
            if actual_answer:
                errors.extend(
                    _require_exact_assistant_answer(
                        messages,
                        actual_answer,
                        "Square lookup",
                    )
                )
        else:
            errors.extend(
                _require_exact_assistant_answer(
                    messages,
                    str(expected_answer),
                    "Square lookup",
                )
            )

    if task == "1.7_rank_lookup":
        expected_answer = metadata.get("expected_answer", "")
        rank = metadata.get("rank", "")
        fen_rank_row = metadata.get("fen_rank_row", "")
        if not expected_answer or not rank or not fen_rank_row:
            errors.append("Missing rank/fen_rank_row/expected_answer metadata for rank lookup")
        elif validate_fen(fen, chess960=is_960):
            try:
                rank_int = int(str(rank))
                board = chess.Board(fen, chess960=is_960)
                actual_row = _fen_rank_row(board, rank_int)
                actual_answer = f"rank {rank_int}: {actual_row}"
            except (TypeError, ValueError):
                actual_row = ""
                actual_answer = ""
                errors.append("Invalid rank metadata for rank lookup")
            if actual_row and str(fen_rank_row) != actual_row:
                errors.append("Rank lookup metadata does not match actual FEN")
            if actual_answer and str(expected_answer) != actual_answer:
                errors.append("Rank lookup expected_answer metadata does not match actual FEN")
            if actual_answer:
                errors.extend(
                    _require_exact_assistant_answer(
                        messages,
                        actual_answer,
                        "Rank lookup",
                    )
                )
        else:
            errors.extend(
                _require_exact_assistant_answer(
                    messages,
                    str(expected_answer),
                    "Rank lookup",
                )
            )

    if task == "1.11_square_coordinates":
        expected_answer = metadata.get("expected_answer", "")
        square = metadata.get("square", "")
        if not expected_answer or not square:
            errors.append("Missing square/expected_answer metadata for square coordinates")
        else:
            try:
                actual_answer = _square_coordinates_answer(str(square))
            except (ValueError, TypeError):
                actual_answer = ""
                errors.append("Invalid square metadata for square coordinates")
            if actual_answer and str(expected_answer) != actual_answer:
                errors.append("Square coordinates expected_answer metadata does not match square")
            if actual_answer:
                errors.extend(
                    _require_exact_assistant_answer(
                        messages,
                        actual_answer,
                        "Square coordinates",
                    )
                )

    if task == "1.12_fen_rank_expansion":
        expected_answer = metadata.get("expected_answer", "")
        rank = metadata.get("rank", "")
        fen_rank_row = metadata.get("fen_rank_row", "")
        if not expected_answer or not rank or not fen_rank_row:
            errors.append(
                "Missing rank/fen_rank_row/expected_answer metadata for FEN rank expansion"
            )
        elif validate_fen(fen, chess960=is_960):
            try:
                rank_int = int(str(rank))
                board = chess.Board(fen, chess960=is_960)
                actual_row = _fen_rank_row(board, rank_int)
                actual_answer = _fen_rank_expansion_answer(rank_int, actual_row)
            except (TypeError, ValueError):
                actual_row = ""
                actual_answer = ""
                errors.append("Invalid rank metadata for FEN rank expansion")
            if actual_row and str(fen_rank_row) != actual_row:
                errors.append("FEN rank expansion metadata does not match actual FEN")
            if actual_answer and str(expected_answer) != actual_answer:
                errors.append(
                    "FEN rank expansion expected_answer metadata does not match actual FEN"
                )
            if actual_answer:
                errors.extend(
                    _require_exact_assistant_answer(
                        messages,
                        actual_answer,
                        "FEN rank expansion",
                    )
                )
        else:
            errors.extend(
                _require_exact_assistant_answer(
                    messages,
                    str(expected_answer),
                    "FEN rank expansion",
                )
            )

    if task == "1.13_fen_rank_cell_edit":
        expected_answer = metadata.get("expected_answer", "")
        rank = metadata.get("rank", "")
        file_name = metadata.get("file", "")
        before_row = metadata.get("before_row", "")
        after_fen = metadata.get("after_fen", "")
        after_row = metadata.get("after_row", "")
        move = metadata.get("move", "")
        if not expected_answer or not rank or not file_name or not before_row or not after_fen:
            errors.append(
                "Missing rank/file/before_row/after_fen/expected_answer metadata "
                "for FEN rank cell edit"
            )
        else:
            actual_answer = ""
            actual_after_row = ""
            if move and validate_fen(fen, chess960=is_960):
                # Derive both rank rows from the move applied to the FEN so
                # multi-edit ranks (castling, en passant) stay truthful.
                try:
                    rank_int = int(str(rank))
                    board_before = chess.Board(fen, chess960=is_960)
                    parsed_move = board_before.parse_uci(str(move))
                    board_after = board_before.copy(stack=False)
                    board_after.push(parsed_move)
                    actual_before_row = _fen_rank_row(board_before, rank_int)
                    actual_after_row = _fen_rank_row(board_after, rank_int)
                    actual_answer = (
                        f"rank {rank_int}: {actual_before_row} -> {actual_after_row}"
                    )
                    cells = _expand_fen_rank_row(actual_after_row)
                    file_index = chess.FILE_NAMES.index(str(file_name))
                    if str(before_row) != actual_before_row:
                        errors.append(
                            "FEN rank cell edit before_row metadata does not match actual FEN"
                        )
                    if cells[file_index] != str(after_fen):
                        errors.append(
                            "FEN rank cell edit after_fen metadata does not match actual FEN"
                        )
                except (TypeError, ValueError, AssertionError):
                    actual_answer = ""
                    actual_after_row = ""
                    errors.append("Invalid metadata for FEN rank cell edit")
            else:
                try:
                    rank_int = int(str(rank))
                    actual_answer = _rank_cell_edit_answer(
                        str(before_row),
                        rank_int,
                        str(file_name),
                        str(after_fen),
                    )
                    actual_after_row = actual_answer.split(" -> ", 1)[1]
                except (TypeError, ValueError):
                    actual_answer = ""
                    actual_after_row = ""
                    errors.append("Invalid metadata for FEN rank cell edit")
            if actual_after_row and after_row and str(after_row) != actual_after_row:
                errors.append("FEN rank cell edit after_row metadata does not match edit")
            if actual_answer and str(expected_answer) != actual_answer:
                errors.append("FEN rank cell edit expected_answer metadata does not match edit")
            if actual_answer:
                errors.extend(
                    _require_exact_assistant_answer(
                        messages,
                        actual_answer,
                        "FEN rank cell edit",
                    )
                )

    if task == "1.14_fen_board_edit":
        expected_answer = metadata.get("expected_answer", "")
        move = metadata.get("move", "")
        board_fen_before = metadata.get("board_fen_before", "")
        board_fen_after = metadata.get("board_fen_after", "")
        if not expected_answer or not board_fen_before or not board_fen_after:
            errors.append(
                "Missing board_fen_before/board_fen_after/expected_answer metadata "
                "for FEN board edit"
            )
        elif validate_fen(fen, chess960=is_960):
            if move and not validate_move_legal(fen, str(move), chess960=is_960):
                errors.append("FEN board edit metadata move is not legal in FEN")
            try:
                board = chess.Board(fen, chess960=is_960)
                actual_before = board.board_fen()
                if move:
                    board.push(board.parse_uci(str(move)))
                actual_after = board.board_fen()
            except (ValueError, TypeError, AssertionError):
                actual_before = ""
                actual_after = ""
                errors.append("FEN board edit metadata move cannot be applied")
            if actual_before and str(board_fen_before) != actual_before:
                errors.append("FEN board edit before metadata does not match actual FEN")
            if actual_after and str(board_fen_after) != actual_after:
                errors.append("FEN board edit after metadata does not match actual FEN")
            actual_answer = _board_edit_answer(str(board_fen_after))
            if str(expected_answer) != actual_answer:
                errors.append("FEN board edit expected_answer metadata does not match board FEN")
            errors.extend(
                _require_exact_assistant_answer(
                    messages,
                    actual_answer,
                    "FEN board edit",
                )
            )
        else:
            errors.extend(
                _require_exact_assistant_answer(
                    messages,
                    str(expected_answer),
                    "FEN board edit",
                )
            )

    if task == "1.8_move_square_edits":
        expected_answer = metadata.get("expected_answer", "")
        move = metadata.get("move", "")
        if not expected_answer or not move:
            errors.append("Missing move/expected_answer metadata for move square edits")
        elif validate_fen(fen, chess960=is_960) and not validate_move_legal(
            fen,
            str(move),
            chess960=is_960,
        ):
            errors.append("Move square edits metadata move is not legal in FEN")
        elif validate_fen(fen, chess960=is_960):
            actual_answer = _move_square_edits_answer(
                fen,
                str(move),
                chess960=is_960,
            )
            if actual_answer is None:
                errors.append("Move square edits metadata move cannot be applied")
            else:
                if str(expected_answer) != actual_answer:
                    errors.append(
                        "Move square edits expected_answer metadata does not match actual FEN"
                    )
                errors.extend(
                    _require_exact_assistant_answer(
                        messages,
                        actual_answer,
                        "Move square edits",
                    )
                )
        else:
            errors.extend(
                _require_exact_assistant_answer(
                    messages,
                    str(expected_answer),
                    "Move square edits",
                )
            )

    if task == "1.9_fen_assembly":
        expected_answer = metadata.get("expected_answer", "")
        move = metadata.get("move", "")
        result_fen = metadata.get("result_fen", "")
        if not expected_answer or not move or not result_fen:
            errors.append(
                "Missing move/result_fen/expected_answer metadata for FEN assembly"
            )
        elif validate_fen(fen, chess960=is_960) and not validate_move_legal(
            fen,
            str(move),
            chess960=is_960,
        ):
            errors.append("FEN assembly metadata move is not legal in FEN")
        elif validate_fen(fen, chess960=is_960):
            actual = _fen_assembly_answer(fen, str(move), chess960=is_960)
            if actual is None:
                errors.append("FEN assembly metadata move cannot be applied")
            else:
                actual_answer, actual_result_fen = actual
                if str(result_fen) != actual_result_fen:
                    errors.append(
                        "FEN assembly result_fen metadata does not match actual FEN"
                    )
                if str(expected_answer) != actual_answer:
                    errors.append(
                        "FEN assembly expected_answer metadata does not match actual FEN"
                    )
                errors.extend(
                    _require_exact_assistant_answer(
                        messages,
                        actual_answer,
                        "FEN assembly",
                    )
                )
        else:
            errors.extend(
                _require_exact_assistant_answer(
                    messages,
                    str(expected_answer),
                    "FEN assembly",
                )
            )

    if task == "1.10_fen_row_application":
        expected_answer = metadata.get("expected_answer", "")
        move = metadata.get("move", "")
        result_fen = metadata.get("result_fen", "")
        if not expected_answer or not move or not result_fen:
            errors.append(
                "Missing move/result_fen/expected_answer metadata for FEN row application"
            )
        elif validate_fen(fen, chess960=is_960) and not validate_move_legal(
            fen,
            str(move),
            chess960=is_960,
        ):
            errors.append("FEN row application metadata move is not legal in FEN")
        elif validate_fen(fen, chess960=is_960):
            actual = _fen_row_application_answer(fen, str(move), chess960=is_960)
            if actual is None:
                errors.append("FEN row application metadata move cannot be applied")
            else:
                actual_answer, actual_result_fen = actual
                if str(result_fen) != actual_result_fen:
                    errors.append(
                        "FEN row application result_fen metadata does not match actual FEN"
                    )
                if str(expected_answer) != actual_answer:
                    errors.append(
                        "FEN row application expected_answer metadata does not match actual FEN"
                    )
                errors.extend(
                    _require_exact_assistant_answer(
                        messages,
                        actual_answer,
                        "FEN row application",
                    )
                )
        else:
            errors.extend(
                _require_exact_assistant_answer(
                    messages,
                    str(expected_answer),
                    "FEN row application",
                )
            )

    if task == "2.1_legal_move_gen" and validate_fen(fen, chess960=is_960):
        for msg in messages:
            if msg["role"] == "assistant" and msg["content"]:
                move_list = _extract_legal_move_answer_moves(msg["content"])
                if not validate_legal_moves(fen, move_list, chess960=is_960):
                    errors.append("Legal move list does not match actual legal moves")

    if task == "2.0_side_piece_inventory" and validate_fen(fen, chess960=is_960):
        board = chess.Board(fen, chess960=is_960)
        actual_answer, actual_inventory = _side_piece_inventory_answer(board)
        expected_answer = metadata.get("expected_answer")
        if expected_answer is not None and str(expected_answer) != actual_answer:
            errors.append("Side piece inventory expected_answer metadata does not match actual FEN")
        expected_side = metadata.get("side_to_move")
        actual_side = "white" if board.turn == chess.WHITE else "black"
        if expected_side is not None and str(expected_side) != actual_side:
            errors.append("Side piece inventory side_to_move metadata does not match actual FEN")
        metadata_inventory = metadata.get("side_piece_inventory")
        if metadata_inventory is not None and metadata_inventory != actual_inventory:
            errors.append("Side piece inventory metadata does not match actual FEN")
        errors.extend(
            _require_exact_assistant_answer(
                messages,
                actual_answer,
                "Side piece inventory",
            )
        )

    if task == "2.2_piece_specific_moves" and validate_fen(fen, chess960=is_960):
        source_square = metadata.get("source_square", "")
        expected_moves_text = str(metadata.get("expected_moves", ""))
        expected_moves = _parse_uci_list_or_no_moves(expected_moves_text)
        if not source_square or not expected_moves_text:
            errors.append("Missing source_square/expected_moves metadata for piece-specific moves")
        else:
            board = chess.Board(fen, chess960=is_960)
            try:
                source_sq = chess.parse_square(str(source_square))
            except ValueError:
                source_sq = None
                errors.append("Invalid source_square metadata for piece-specific moves")
            if source_sq is not None:
                actual_moves = sorted(
                    move.uci() for move in board.legal_moves if move.from_square == source_sq
                )
                if expected_moves != actual_moves:
                    errors.append("Piece-specific metadata does not match actual legal moves")
                for content in _assistant_contents(messages):
                    if _parse_uci_list_or_no_moves(content) != actual_moves:
                        errors.append("Piece-specific move list does not match actual legal moves")

    if task in {
        "2.6_piece_pseudo_legal_moves",
        "2.7_piece_legal_filter",
        "2.8_king_safety_filter",
        "2.9_legal_moves_by_piece",
    } and validate_fen(fen, chess960=is_960):
        board = chess.Board(fen, chess960=is_960)
        actual_answer = _legal_decomposition_answer(task, board, metadata)
        expected_answer = metadata.get("expected_answer", "")
        if actual_answer is None:
            errors.append("Missing or invalid metadata for legal decomposition")
        else:
            if not expected_answer:
                errors.append("Missing expected_answer metadata for legal decomposition")
            elif str(expected_answer) != actual_answer:
                errors.append("Legal decomposition expected_answer metadata does not match actual FEN")
            errors.extend(
                _require_exact_assistant_answer(
                    messages,
                    actual_answer,
                    "Legal decomposition",
                )
            )

    if task == "2.10_ray_walk" and validate_fen(fen, chess960=is_960):
        board = chess.Board(fen, chess960=is_960)
        source_square = metadata.get("source_square") or metadata.get("square")
        actual_answer = None
        if not source_square:
            errors.append("Missing source_square metadata for ray walk")
        else:
            try:
                square = chess.parse_square(str(source_square))
            except ValueError:
                square = None
                errors.append("Invalid source_square metadata for ray walk")
            if square is not None:
                piece = board.piece_at(square)
                if (
                    piece is None
                    or piece.color != board.turn
                    or piece.piece_type not in SLIDER_RAY_DIRECTIONS
                ):
                    errors.append("Ray walk source square is not a side-to-move slider")
                else:
                    actual_answer = format_ray_walk_answer(board, square)
        if actual_answer is not None:
            expected_answer = metadata.get("expected_answer", "")
            if not expected_answer:
                errors.append("Missing expected_answer metadata for ray walk")
            elif str(expected_answer) != actual_answer:
                errors.append("Ray walk expected_answer metadata does not match actual FEN")
            errors.extend(
                _require_exact_assistant_answer(
                    messages,
                    actual_answer,
                    "Ray walk",
                )
            )

    if task == "2.11_legal_filter_trace" and validate_fen(fen, chess960=is_960):
        board = chess.Board(fen, chess960=is_960)
        actual_answer = format_legal_filter_trace_answer(board)
        if actual_answer is None:
            errors.append("Legal filter trace position exceeds the trace caps")
        else:
            expected_answer = metadata.get("expected_answer", "")
            if not expected_answer:
                errors.append("Missing expected_answer metadata for legal filter trace")
            elif str(expected_answer) != actual_answer:
                errors.append(
                    "Legal filter trace expected_answer metadata does not match actual FEN"
                )
            metadata_moves = metadata.get("legal_moves")
            actual_moves = _move_text(sorted(move.uci() for move in board.legal_moves))
            if metadata_moves is not None and str(metadata_moves) != actual_moves:
                errors.append(
                    "Legal filter trace legal_moves metadata does not match actual FEN"
                )
            errors.extend(
                _require_exact_assistant_answer(
                    messages,
                    actual_answer,
                    "Legal filter trace",
                )
            )

    if task == "2.3_move_legality_check" and validate_fen(fen, chess960=is_960):
        tested_move = metadata.get("tested_move", "")
        if not tested_move:
            errors.append("Missing tested_move metadata for move legality check")
        else:
            board = chess.Board(fen, chess960=is_960)
            classification = classify_move_legality(board, str(tested_move))
            is_legal = classification.is_legal
            expected_is_legal = metadata.get("expected_is_legal")
            if (
                expected_is_legal is not None
                and bool(expected_is_legal) != classification.is_legal
            ):
                errors.append("Move legality expected_is_legal metadata does not match actual FEN")
            reason_label = metadata.get("legality_reason_label")
            if not reason_label:
                errors.append("Missing legality_reason_label metadata for move legality check")
            elif reason_label not in LEGALITY_REASON_LABELS:
                errors.append("Unknown legality_reason_label metadata for move legality check")
            elif reason_label != classification.reason_label:
                errors.append("Move legality reason metadata does not match actual FEN")
            for content in _assistant_contents(messages):
                says_legal = parse_legality_yes_no_answer(content)
                if says_legal is None:
                    errors.append("Move legality answer must be a clear yes/no response")
                    continue
                if says_legal != is_legal:
                    errors.append(
                        f"Answer says {'legal' if says_legal else 'illegal'} "
                        f"but move {tested_move!r} is actually "
                        f"{'legal' if is_legal else 'illegal'}"
                    )
                answer_reason = parse_legality_reason_label(content)
                if answer_reason is None:
                    errors.append("Move legality answer must include a recognized Reason label")
                elif answer_reason != classification.reason_label:
                    errors.append("Move legality reason in assistant answer does not match actual FEN")

    if task == "2.4_check_detection" and validate_fen(fen, chess960=is_960):
        board = chess.Board(fen, chess960=is_960)
        actual_label, actual_answer = _check_state_label(board)
        state_label = metadata.get("state_label")
        if state_label is not None and state_label != actual_label:
            errors.append("Check detection metadata does not match actual board state")
        errors.extend(
            _require_exact_assistant_answer(
                messages,
                actual_answer,
                "Check detection",
            )
        )

    if task == "2.5_special_rules":
        expected_answer = metadata.get("expected_answer", "")
        if not expected_answer:
            errors.append("Missing expected_answer metadata for special rules")
        else:
            errors.extend(
                _require_exact_assistant_answer(
                    messages,
                    str(expected_answer),
                    "Special rules",
                )
            )

    if task in {
        "1.5_state_tracking",
        "1.19_multi_move_state_tracking",
    } and validate_fen(fen, chess960=is_960):
        result_fen = metadata.get("result_fen", "")
        moves_str = metadata.get("moves", "")
        if not result_fen or not moves_str:
            errors.append("Missing state tracking metadata: result_fen and moves are required")
        else:
            move_list = str(moves_str).split()
            if task == "1.19_multi_move_state_tracking" and len(move_list) < 2:
                errors.append(
                    "Multi-move state tracking requires at least 2 moves"
                )
            if not validate_state_tracking(
                fen,
                move_list,
                result_fen,
                chess960=is_960,
            ):
                errors.append("State tracking: applying moves does not produce result FEN")
            for content in _assistant_contents(messages):
                if not _state_tracking_answer_matches_result(content, str(result_fen)):
                    errors.append("Assistant state tracking answer does not match result FEN")

    return (len(errors) == 0, errors)


__all__ = [
    "validate_example",
    "validate_fen",
    "validate_legal_moves",
    "validate_move_legal",
    "validate_state_tracking",
    "validate_template_complete",
    "validate_think_move_format",
]
