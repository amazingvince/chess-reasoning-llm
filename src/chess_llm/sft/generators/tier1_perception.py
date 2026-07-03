"""Tier 1: Perception and state tasks (~1.32M examples).

1.1 FEN -> ASCII board diagram
1.2 ASCII board -> FEN
1.3 Piece identification (what's on a square / where are the pieces)
1.4 Piece counting (material count, balance)
1.5 State tracking (apply moves, report resulting position)
1.6 Square lookup (read a single square from FEN)
1.7 Rank lookup (read one FEN rank row)
1.8 Move square edits (single-move lookup/edit trace)
1.9 FEN assembly (apply one move and assemble the full resulting FEN)
1.10 FEN row application (rewrite affected compressed rank rows)
1.11 Square coordinates (map square to FEN row/file cell)
1.12 FEN rank expansion (expand compressed rank row into file cells)
1.13 FEN rank cell edit (rewrite one compressed rank row cell)
1.14 FEN board edit (apply changed square edits to board FEN)
1.19 Multi-move state tracking (apply 2-3 moves, report resulting FEN)
"""

from __future__ import annotations

from random import Random
from typing import Iterator

import chess

from chess_llm.formats import render_ascii_board
from chess_llm.sft.context import board_from_raw
from chess_llm.sft.generators.base import TaskGenerator
from chess_llm.sft.templates import TEMPLATES, select_template


_board_to_ascii = render_ascii_board


_PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}
_PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}
_PIECE_INVENTORY_ORDER = (
    chess.KING,
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
    chess.PAWN,
)
_PIECE_INVENTORY_LABELS = {
    chess.KING: "king",
    chess.QUEEN: "queens",
    chess.ROOK: "rooks",
    chess.BISHOP: "bishops",
    chess.KNIGHT: "knights",
    chess.PAWN: "pawns",
}
_DEFAULT_STATE_TRACKING_MIN_PLIES = 1
_DEFAULT_STATE_TRACKING_MAX_PLIES = 1


def _shuffled_fen_pool(config: dict, rng: Random) -> list:
    """Return a locally shuffled copy of the shared fen_pool prefix."""
    pool = list(config.get("fen_pool", []))
    shuffle = getattr(rng, "shuffle", None)
    if callable(shuffle):
        shuffle(pool)
    return pool


def _append_full_fen_state(user_text: str, context: dict) -> str:
    """Append non-board FEN fields so full-FEN targets are inferable."""
    state_lines = [
        "State needed for full FEN:",
        f"Side to move: {context['side_to_move']}",
        f"Castling rights: {context['castling_rights']}",
        f"En passant: {context['en_passant_square']}",
        f"Halfmove clock: {context['halfmove_clock']}",
        f"Fullmove number: {context['fullmove_number']}",
    ]
    return f"{user_text}\n" + "\n".join(state_lines)


def _state_tracking_ply_bounds(
    config: dict,
    *,
    min_key: str = "state_tracking_min_plies",
    max_key: str = "state_tracking_max_plies",
    default_min: int = _DEFAULT_STATE_TRACKING_MIN_PLIES,
    default_max: int = _DEFAULT_STATE_TRACKING_MAX_PLIES,
) -> tuple[int, int]:
    """Return configured state-tracking move bounds with Phase A-safe defaults."""
    min_plies = int(config.get(min_key, default_min))
    max_plies = int(config.get(max_key, default_max))
    min_plies = max(1, min_plies)
    max_plies = max(min_plies, max_plies)
    return min_plies, max_plies


def _piece_phrase(piece: chess.Piece | None) -> str:
    if piece is None:
        return "empty"
    color = "white" if piece.color == chess.WHITE else "black"
    return f"{color} {_PIECE_NAMES[piece.piece_type]}"


def _piece_fen_char(piece: chess.Piece | None) -> str:
    if piece is None:
        return "1"
    return piece.symbol()


def _piece_inventory_for_color(board: chess.Board, color: bool) -> str:
    parts: list[str] = []
    for piece_type in _PIECE_INVENTORY_ORDER:
        squares = [
            chess.square_name(square)
            for square in chess.SQUARES
            if (
                (piece := board.piece_at(square)) is not None
                and piece.color == color
                and piece.piece_type == piece_type
            )
        ]
        square_text = ",".join(squares) if squares else "none"
        parts.append(f"{_PIECE_INVENTORY_LABELS[piece_type]}={square_text}")
    return "; ".join(parts)


def _piece_count_vector(counts: dict[str, int]) -> dict[str, int]:
    return {
        "king": counts.get("king", 0),
        "queen": counts.get("queen", 0),
        "rook": counts.get("rook", 0),
        "bishop": counts.get("bishop", 0),
        "knight": counts.get("knight", 0),
        "pawn": counts.get("pawn", 0),
    }


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
    for piece_type in _PIECE_INVENTORY_ORDER:
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
        for piece_type in _PIECE_INVENTORY_ORDER
    )


def _material_value_vector_text(summary: dict[str, object]) -> str:
    values = summary["values"]
    assert isinstance(values, dict)
    total = int(summary["total"])
    parts = [
        f"{_PIECE_NAMES[piece_type]}={values.get(_PIECE_NAMES[piece_type], 0)}"
        for piece_type in _PIECE_INVENTORY_ORDER
    ]
    parts.append(f"total={total}")
    return "; ".join(parts)


def _material_balance_sentence(white_total: int, black_total: int) -> str:
    balance = white_total - black_total
    if balance > 0:
        return f"White is up {balance} point(s) of material."
    if balance < 0:
        return f"Black is up {abs(balance)} point(s) of material."
    return "Material is equal."


def _format_material_inventory_answer(board: chess.Board) -> str:
    summary = _material_summary(board)
    return (
        f"White inventory: {_material_inventory_vector(summary['white'])}.\n"
        f"Black inventory: {_material_inventory_vector(summary['black'])}."
    )


def _format_material_piece_counts_answer(board: chess.Board) -> str:
    summary = _material_summary(board)
    return (
        f"White counts: {_material_count_vector_text(summary['white'])}.\n"
        f"Black counts: {_material_count_vector_text(summary['black'])}."
    )


def _format_material_value_totals_answer(board: chess.Board) -> str:
    summary = _material_summary(board)
    return (
        f"White values: {_material_value_vector_text(summary['white'])}.\n"
        f"Black values: {_material_value_vector_text(summary['black'])}."
    )


def _format_material_balance_trace_answer(board: chess.Board) -> str:
    summary = _material_summary(board)
    white = summary["white"]
    black = summary["black"]
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


def _state_tracking_changed_squares(
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
        changed.append({
            "square": chess.square_name(square),
            "before": _piece_phrase(before_piece),
            "after": _piece_phrase(after_piece),
            "before_fen": _piece_fen_char(before_piece),
            "after_fen": _piece_fen_char(after_piece),
        })
    if preferred_order:
        rank = {square: index for index, square in enumerate(preferred_order)}
        changed.sort(
            key=lambda item: (
                rank.get(chess.parse_square(item["square"]), len(rank)),
                item["square"],
            )
        )
    return changed


def _state_tracking_rank_updates(
    before: chess.Board,
    after: chess.Board,
    changed_squares: list[dict[str, str]],
) -> list[dict[str, str]]:
    before_rows = before.board_fen().split("/")
    after_rows = after.board_fen().split("/")
    affected_ranks = []
    for item in changed_squares:
        rank = chess.square_rank(chess.parse_square(item["square"]))
        if rank not in affected_ranks:
            affected_ranks.append(rank)

    updates = []
    for rank in affected_ranks:
        row_index = 7 - rank
        updates.append({
            "rank": str(rank + 1),
            "before": before_rows[row_index],
            "after": after_rows[row_index],
        })
    return updates


def _fen_rank_row(board: chess.Board, rank: int) -> str:
    """Return the compressed FEN row for a 1-based board rank."""
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


def _format_fen_rank_expansion_answer(rank: int, row: str) -> str:
    cells = _expand_fen_rank_row(row)
    assignments = [
        f"{file_name}={cell}"
        for file_name, cell in zip(chess.FILE_NAMES, cells, strict=True)
    ]
    return f"rank {rank}: {'; '.join(assignments)}"


def _format_rank_cell_edit_answer(rank: str, before_row: str, after_row: str) -> str:
    return f"rank {rank}: {before_row} -> {after_row}"


def _format_board_edit_text(changed_squares: list[dict[str, str]]) -> str:
    edits = [
        f"{item['square']} {item['before_fen']}->{item['after_fen']}"
        for item in changed_squares
    ]
    return "; ".join(edits)


def _format_fen_board_edit_answer(board_fen_after: str) -> str:
    return f"Result board FEN: {board_fen_after}"


def _format_move_square_edits_answer(detail: dict[str, object]) -> str:
    changed_squares = [
        item
        for item in detail.get("changed_squares", [])
        if isinstance(item, dict)
    ]
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
        square = chess.parse_square(str(item["square"]))
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


def _format_fen_row_application_answer(
    detail: dict[str, object],
    result_fen: str,
) -> str:
    rank_updates = [
        item
        for item in detail.get("rank_updates", [])
        if isinstance(item, dict)
    ]
    rewrites = [
        f"rank {item['rank']} {item['before']}->{item['after']}"
        for item in rank_updates
    ]
    return "\n".join(
        [
            f"Rows: {'; '.join(rewrites)}.",
            f"Result FEN: {result_fen}",
        ]
    )


def _state_tracking_move_kind(board: chess.Board, move: chess.Move) -> str:
    moving_piece = board.piece_at(move.from_square)
    if board.is_castling(move):
        return "castling"
    if board.is_en_passant(move):
        return "en_passant"
    if move.promotion and board.is_capture(move):
        return "promotion_capture"
    if move.promotion:
        return "promotion"
    if board.is_capture(move):
        return "capture"
    if (
        moving_piece is not None
        and moving_piece.piece_type == chess.PAWN
        and abs(chess.square_rank(move.to_square) - chess.square_rank(move.from_square)) == 2
    ):
        return "pawn_double_push"
    return "quiet"


def _state_tracking_captured_piece(
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


def _state_tracking_move_detail(
    before: chess.Board,
    move: chess.Move,
    after: chess.Board,
    move_number: int,
) -> dict[str, object]:
    piece = before.piece_at(move.from_square)
    captured_piece, captured_square = _state_tracking_captured_piece(before, move)
    kind = _state_tracking_move_kind(before, move)
    from_square = chess.square_name(move.from_square)
    to_square = chess.square_name(move.to_square)
    changed_squares = _state_tracking_changed_squares(
        before,
        after,
        preferred_order=[move.from_square, move.to_square],
    )
    rank_updates = _state_tracking_rank_updates(before, after, changed_squares)

    if kind == "castling":
        side = "kingside" if chess.square_file(move.to_square) > chess.square_file(move.from_square) else "queenside"
        sentence = f"Move {move_number}: {_piece_phrase(piece)} {move.uci()} ({side} castling)."
    else:
        sentence = f"Move {move_number}: {_piece_phrase(piece)} {move.uci()}."

    if captured_piece is not None:
        sentence += f" Captures {_piece_phrase(captured_piece)} on {captured_square}."
    if move.promotion:
        promoted_piece = chess.Piece(move.promotion, piece.color if piece else before.turn)
        sentence += f" Promotes to {_piece_phrase(promoted_piece)}."

    return {
        "move": move.uci(),
        "move_kind": kind,
        "piece": _piece_phrase(piece),
        "from": from_square,
        "to": to_square,
        "captured": _piece_phrase(captured_piece) if captured_piece else "",
        "captured_square": captured_square,
        "promotion": _PIECE_NAMES.get(move.promotion, "") if move.promotion else "",
        "changed_squares": changed_squares,
        "rank_updates": rank_updates,
        "description": sentence,
    }


def _format_state_tracking_answer(
    move_details: list[dict[str, object]],
    result_fen: str,
) -> str:
    lines = []
    for detail in move_details:
        lines.append(str(detail["description"]))
        lines.extend(_format_move_square_edits_answer(detail).splitlines())
    lines.append(f"Result FEN: {result_fen}")
    return "\n".join(lines)


class FENToBoard(TaskGenerator):
    """Task 1.1: FEN -> ASCII board diagram."""

    def task_id(self) -> str:
        return "1.1_fen_to_board"

    def tier(self) -> int:
        return 1

    def generate(self) -> Iterator[dict]:
        pool = _shuffled_fen_pool(self.config, self.rng)
        target = self.target_volume()
        count = 0
        for entry in pool:
            if count >= target:
                return
            raw = self.source_row(entry)
            if self.is_blocked(raw):
                continue

            board = board_from_raw(raw)
            if board is None:
                continue
            ascii_board = _board_to_ascii(board)

            raw["board"] = ascii_board
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=ascii_board)
            count += 1


class BoardToFEN(TaskGenerator):
    """Task 1.2: ASCII board -> FEN."""

    def task_id(self) -> str:
        return "1.2_board_to_fen"

    def tier(self) -> int:
        return 1

    def generate(self) -> Iterator[dict]:
        pool = _shuffled_fen_pool(self.config, self.rng)
        target = self.target_volume()
        count = 0
        for entry in pool:
            if count >= target:
                return
            raw = self.source_row(entry)
            if self.is_blocked(raw):
                continue

            board = board_from_raw(raw)
            if board is None:
                continue
            fen = board.fen()
            raw["fen"] = fen
            ascii_board = _board_to_ascii(board)

            raw["board"] = ascii_board
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            user_text = _append_full_fen_state(
                user_text,
                self.build_template_context(raw),
            )
            yield self.format_example(raw, template_text=user_text, assistant_content=fen)
            count += 1


class PieceIdentification(TaskGenerator):
    """Task 1.3: Identify pieces on squares or locate specific pieces."""

    def task_id(self) -> str:
        return "1.3_piece_identification"

    def tier(self) -> int:
        return 1

    def generate(self) -> Iterator[dict]:
        pool = _shuffled_fen_pool(self.config, self.rng)
        target = self.target_volume()
        count = 0
        square_query_count = 0
        for entry in pool:
            if count >= target:
                return
            raw_base = self.source_row(entry)
            if self.is_blocked(raw_base):
                continue
            fen = raw_base["fen"]
            board = board_from_raw(raw_base)
            if board is None:
                continue

            # Randomly choose subtask: what's on square, or where are pieces
            if self.rng.random() < 0.5:
                # What is on a specific square?
                sq = _select_piece_identification_square(
                    board,
                    prefer_occupied=(square_query_count % 2 == 0),
                    rng=self.rng,
                )
                square_query_count += 1
                square_name = chess.square_name(sq)
                piece = board.piece_at(sq)
                if piece:
                    answer = f"{('white' if piece.color == chess.WHITE else 'black')} {_PIECE_NAMES[piece.piece_type]}"
                else:
                    answer = "empty"

                metadata = dict(raw_base.get("metadata", {}))
                metadata.update(
                    {
                        "query_kind": "square_piece",
                        "square": square_name,
                        "square_has_piece": piece is not None,
                        "expected_answer": answer,
                    }
                )
                raw = {
                    **raw_base,
                    "square": square_name,
                    "color": "",
                    "piece": "",
                    "metadata": metadata,
                }
                templates_with_sq = [
                    t for t in TEMPLATES[self.task_id()]
                    if "{square}" in t and "{color}" not in t and "{piece}" not in t
                ]
                user_text = self.render_template(raw, self.rng.choice(templates_with_sq))
                yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            else:
                # Where are specific pieces?
                color = self.rng.choice([chess.WHITE, chess.BLACK])
                color_name = "white" if color == chess.WHITE else "black"
                # Pick a piece type that exists
                piece_types = []
                for sq in chess.SQUARES:
                    p = board.piece_at(sq)
                    if p and p.color == color:
                        piece_types.append(p.piece_type)
                if not piece_types:
                    continue
                pt = self.rng.choice(list(set(piece_types)))
                piece_name = _PIECE_NAMES[pt]

                squares = []
                for sq in chess.SQUARES:
                    p = board.piece_at(sq)
                    if p and p.color == color and p.piece_type == pt:
                        squares.append(chess.square_name(sq))
                answer = " ".join(sorted(squares))

                metadata = dict(raw_base.get("metadata", {}))
                metadata.update(
                    {
                        "query_kind": "locate_pieces",
                        "color": color_name,
                        "piece": piece_name,
                        "expected_answer": answer,
                    }
                )
                raw = {
                    **raw_base,
                    "color": color_name,
                    "piece": piece_name,
                    "square": "",
                    "metadata": metadata,
                }
                templates_for_locate = [
                    t for t in TEMPLATES[self.task_id()]
                    if "{color}" in t and "{piece}" in t and "{square}" not in t
                ]
                user_text = self.render_template(raw, self.rng.choice(templates_for_locate))
                yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


def _select_piece_identification_square(
    board: chess.Board,
    *,
    prefer_occupied: bool,
    rng: Random,
) -> int:
    """Select a square while keeping occupied/empty labels useful in SFT data."""
    occupied = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
    empty = [sq for sq in chess.SQUARES if board.piece_at(sq) is None]

    primary = occupied if prefer_occupied else empty
    fallback = empty if prefer_occupied else occupied
    candidates = primary or fallback or list(chess.SQUARES)
    return rng.choice(candidates)


def _piece_counting_template_kind(template: str) -> str:
    lower = template.lower()
    if "how many pieces does {color}" in lower:
        return "color_total"
    if "how many {piece}s" in lower or "count the {piece}s" in lower:
        return "piece_type"
    if "minor pieces" in lower:
        return "minor_pieces"
    return "full_material"


def _select_piece_counting_template(config: dict, rng: Random) -> str:
    template = select_template("1.4_piece_counting", rng)
    if config.get("piece_counting_include_partial", False):
        return template
    if _piece_counting_template_kind(template) == "full_material":
        return template

    full_templates = [
        candidate
        for candidate in TEMPLATES["1.4_piece_counting"]
        if _piece_counting_template_kind(candidate) == "full_material"
    ]
    return rng.choice(full_templates or [template])


class PieceCounting(TaskGenerator):
    """Task 1.4: Count pieces, material balance."""

    def task_id(self) -> str:
        return "1.4_piece_counting"

    def tier(self) -> int:
        return 1

    def generate(self) -> Iterator[dict]:
        pool = _shuffled_fen_pool(self.config, self.rng)
        target = self.target_volume()
        count = 0
        for entry in pool:
            if count >= target:
                return
            raw = self.source_row(entry)
            if self.is_blocked(raw):
                continue
            board = board_from_raw(raw)
            if board is None:
                continue
            fen = raw["fen"]

            # Count pieces per side
            white_counts: dict[str, int] = {}
            black_counts: dict[str, int] = {}
            white_total = 0
            black_total = 0
            white_material = 0
            black_material = 0

            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece is None:
                    continue
                name = _PIECE_NAMES[piece.piece_type]
                val = _PIECE_VALUES[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_counts[name] = white_counts.get(name, 0) + 1
                    white_total += 1
                    white_material += val
                else:
                    black_counts[name] = black_counts.get(name, 0) + 1
                    black_total += 1
                    black_material += val

            # Format answer
            w_parts = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in sorted(white_counts.items())]
            b_parts = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in sorted(black_counts.items())]
            balance = white_material - black_material
            if balance > 0:
                balance_str = f"White is up {balance} point(s) of material."
            elif balance < 0:
                balance_str = f"Black is up {abs(balance)} point(s) of material."
            else:
                balance_str = "Material is equal."
            material_count_answer = (
                f"White ({white_total} pieces): {', '.join(w_parts)}. "
                f"Black ({black_total} pieces): {', '.join(b_parts)}. "
                f"{balance_str}"
            )

            tpl = _select_piece_counting_template(self.config, self.rng)
            tpl_lower = tpl.lower()
            color = self.rng.choice(["white", "black"])
            if "how many {piece}s" in tpl_lower or "count the {piece}s" in tpl_lower:
                piece_type_name = self.rng.choice(
                    ["pawn", "knight", "bishop", "rook", "queen"]
                )
            else:
                piece_type_name = self.rng.choice(
                    ["pawn", "knight", "bishop", "rook", "queen", "piece"]
                )
            # Only expose color/piece when the template uses them so the
            # example identity is stable for templates that use neither.
            if "{color}" in tpl:
                raw["color"] = color
            if "{piece}" in tpl:
                raw["piece"] = piece_type_name
            user_text = self.render_template(raw, tpl)

            # Branch the answer based on the selected template
            metadata = dict(raw.get("metadata", {}))
            if "how many pieces does {color}" in tpl_lower:
                # Answer only for the requested color
                if color == "white":
                    answer = f"White has {white_total} piece(s): {', '.join(w_parts)}."
                else:
                    answer = f"Black has {black_total} piece(s): {', '.join(b_parts)}."
                metadata.update({"count_kind": "color_total", "color": color})
            elif "how many {piece}s" in tpl_lower or "count the {piece}s" in tpl_lower:
                # Count only the requested piece type
                _name_to_type = {
                    "pawn": chess.PAWN, "knight": chess.KNIGHT,
                    "bishop": chess.BISHOP, "rook": chess.ROOK, "queen": chess.QUEEN,
                }
                pt = _name_to_type.get(piece_type_name)
                if pt is None:
                    continue
                wc = white_counts.get(piece_type_name, 0)
                bc = black_counts.get(piece_type_name, 0)
                answer = (
                    f"White has {wc} {piece_type_name}(s), "
                    f"black has {bc} {piece_type_name}(s). "
                    f"Total: {wc + bc}."
                )
                metadata.update({"count_kind": "piece_type", "piece": piece_type_name})
            elif "minor pieces" in tpl_lower:
                # Count only knights and bishops
                w_minor = white_counts.get("knight", 0) + white_counts.get("bishop", 0)
                b_minor = black_counts.get("knight", 0) + black_counts.get("bishop", 0)
                if color == "white":
                    answer = f"White has {w_minor} minor piece(s) ({white_counts.get('knight', 0)} knight(s), {white_counts.get('bishop', 0)} bishop(s))."
                else:
                    answer = f"Black has {b_minor} minor piece(s) ({black_counts.get('knight', 0)} knight(s), {black_counts.get('bishop', 0)} bishop(s))."
                metadata.update({"count_kind": "minor_pieces", "color": color})
            else:
                # Full material count for both sides (generic templates)
                white_inventory = _piece_inventory_for_color(board, chess.WHITE)
                black_inventory = _piece_inventory_for_color(board, chess.BLACK)
                answer = material_count_answer
                metadata.update(
                    {
                        "count_kind": "full_material",
                        "white_inventory": white_inventory,
                        "black_inventory": black_inventory,
                        "white_counts": _piece_count_vector(white_counts),
                        "black_counts": _piece_count_vector(black_counts),
                        "white_total": white_total,
                        "black_total": black_total,
                        "white_material": white_material,
                        "black_material": black_material,
                        "material_balance": balance,
                        "material_count_answer": material_count_answer,
                    }
                )

            metadata["expected_answer"] = answer
            raw["metadata"] = metadata
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class _MaterialDecompositionTask(TaskGenerator):
    """Base for compact material-count decomposition tasks."""

    query_kind = "material_decomposition"

    def tier(self) -> int:
        return 1

    def answer_for_board(self, board: chess.Board) -> str:
        raise NotImplementedError

    def generate(self) -> Iterator[dict]:
        pool = _shuffled_fen_pool(self.config, self.rng)
        target = self.target_volume()
        count = 0
        for entry in pool:
            if count >= target:
                return
            raw = self.source_row(entry)
            if self.is_blocked(raw):
                continue
            board = board_from_raw(raw)
            if board is None:
                continue

            answer = self.answer_for_board(board)
            metadata = dict(raw.get("metadata", {}))
            metadata.update(
                {
                    "query_kind": self.query_kind,
                    "expected_answer": answer,
                }
            )
            raw["metadata"] = metadata
            user_text = self.render_template(raw)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class MaterialInventory(_MaterialDecompositionTask):
    """Task 1.15: List material inventory by side and piece type."""

    query_kind = "material_inventory"

    def task_id(self) -> str:
        return "1.15_material_inventory"

    def answer_for_board(self, board: chess.Board) -> str:
        return _format_material_inventory_answer(board)


class MaterialPieceCounts(_MaterialDecompositionTask):
    """Task 1.16: Count each piece type for both sides."""

    query_kind = "material_piece_counts"

    def task_id(self) -> str:
        return "1.16_material_piece_counts"

    def answer_for_board(self, board: chess.Board) -> str:
        return _format_material_piece_counts_answer(board)


class MaterialValueTotals(_MaterialDecompositionTask):
    """Task 1.17: Convert piece counts into material value totals."""

    query_kind = "material_value_totals"

    def task_id(self) -> str:
        return "1.17_material_value_totals"

    def answer_for_board(self, board: chess.Board) -> str:
        return _format_material_value_totals_answer(board)


class MaterialBalanceTrace(_MaterialDecompositionTask):
    """Task 1.18: Structured material trace ending in the balance sentence."""

    query_kind = "material_balance_trace"

    def task_id(self) -> str:
        return "1.18_material_balance_trace"

    def answer_for_board(self, board: chess.Board) -> str:
        return _format_material_balance_trace_answer(board)


def _select_square_lookup_square(
    board: chess.Board,
    *,
    prefer_occupied: bool,
    rng: Random,
) -> int:
    occupied = [
        sq
        for sq in chess.SQUARES
        if board.piece_at(sq) is not None
        and board.piece_at(sq).piece_type != chess.KING
    ]
    occupied_fallback = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
    empty = [sq for sq in chess.SQUARES if board.piece_at(sq) is None]

    primary = (occupied or occupied_fallback) if prefer_occupied else empty
    fallback = empty if prefer_occupied else (occupied or occupied_fallback)
    candidates = primary or fallback or list(chess.SQUARES)
    return rng.choice(candidates)


class SquareLookup(TaskGenerator):
    """Task 1.6: Read one square directly from FEN."""

    def task_id(self) -> str:
        return "1.6_square_lookup"

    def tier(self) -> int:
        return 1

    def generate(self) -> Iterator[dict]:
        pool = _shuffled_fen_pool(self.config, self.rng)
        target = self.target_volume()
        count = 0
        for entry in pool:
            if count >= target:
                return
            raw_base = self.source_row(entry)
            if self.is_blocked(raw_base):
                continue
            board = board_from_raw(raw_base)
            if board is None:
                continue

            square = _select_square_lookup_square(
                board,
                prefer_occupied=(count % 2 == 0),
                rng=self.rng,
            )
            square_name = chess.square_name(square)
            answer = f"{square_name}={_piece_phrase(board.piece_at(square))}"
            metadata = dict(raw_base.get("metadata", {}))
            metadata.update(
                {
                    "square": square_name,
                    "expected_answer": answer,
                    "query_kind": "square_lookup",
                    "square_has_piece": board.piece_at(square) is not None,
                }
            )
            raw = {
                **raw_base,
                "square": square_name,
                "metadata": metadata,
            }
            user_text = self.render_template(raw)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class RankLookup(TaskGenerator):
    """Task 1.7: Read one compressed rank row from FEN."""

    def task_id(self) -> str:
        return "1.7_rank_lookup"

    def tier(self) -> int:
        return 1

    def generate(self) -> Iterator[dict]:
        pool = _shuffled_fen_pool(self.config, self.rng)
        target = self.target_volume()
        count = 0
        for entry in pool:
            if count >= target:
                return
            raw_base = self.source_row(entry)
            if self.is_blocked(raw_base):
                continue
            board = board_from_raw(raw_base)
            if board is None:
                continue

            rank = self.rng.randint(1, 8)
            row = _fen_rank_row(board, rank)
            answer = f"rank {rank}: {row}"
            metadata = dict(raw_base.get("metadata", {}))
            metadata.update(
                {
                    "rank": str(rank),
                    "fen_rank_row": row,
                    "expected_answer": answer,
                    "query_kind": "rank_lookup",
                }
            )
            raw = {
                **raw_base,
                "rank": str(rank),
                "metadata": metadata,
            }
            user_text = self.render_template(raw)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class SquareCoordinates(TaskGenerator):
    """Task 1.11: Map a square to its FEN row/file coordinates."""

    def task_id(self) -> str:
        return "1.11_square_coordinates"

    def tier(self) -> int:
        return 1

    def generate(self) -> Iterator[dict]:
        pool = _shuffled_fen_pool(self.config, self.rng)
        target = self.target_volume()
        count = 0
        for entry in pool:
            if count >= target:
                return
            raw_base = self.source_row(entry)
            if self.is_blocked(raw_base):
                continue
            board = board_from_raw(raw_base)
            if board is None:
                continue

            square = _select_square_lookup_square(
                board,
                prefer_occupied=(count % 2 == 0),
                rng=self.rng,
            )
            square_name = chess.square_name(square)
            answer = _square_coordinates_answer(square_name)
            metadata = dict(raw_base.get("metadata", {}))
            metadata.update(
                {
                    "square": square_name,
                    "expected_answer": answer,
                    "query_kind": "square_coordinates",
                    "file": chess.FILE_NAMES[chess.square_file(square)],
                    "rank": str(chess.square_rank(square) + 1),
                    "fen_row_from_top": 8 - chess.square_rank(square),
                    "file_index": chess.square_file(square) + 1,
                }
            )
            raw = {
                **raw_base,
                "square": square_name,
                "metadata": metadata,
            }
            user_text = self.render_template(raw)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class FENRankExpansion(TaskGenerator):
    """Task 1.12: Expand a compressed FEN rank row into file cells."""

    def task_id(self) -> str:
        return "1.12_fen_rank_expansion"

    def tier(self) -> int:
        return 1

    def generate(self) -> Iterator[dict]:
        pool = _shuffled_fen_pool(self.config, self.rng)
        target = self.target_volume()
        count = 0
        for entry in pool:
            if count >= target:
                return
            raw_base = self.source_row(entry)
            if self.is_blocked(raw_base):
                continue
            board = board_from_raw(raw_base)
            if board is None:
                continue

            rank = self.rng.randint(1, 8)
            row = _fen_rank_row(board, rank)
            answer = _format_fen_rank_expansion_answer(rank, row)
            metadata = dict(raw_base.get("metadata", {}))
            metadata.update(
                {
                    "rank": str(rank),
                    "fen_rank_row": row,
                    "expanded_cells": _expand_fen_rank_row(row),
                    "expected_answer": answer,
                    "query_kind": "fen_rank_expansion",
                }
            )
            raw = {
                **raw_base,
                "rank": str(rank),
                "fen_rank_row": row,
                "metadata": metadata,
            }
            user_text = self.render_template(raw)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class FENRankCellEdit(TaskGenerator):
    """Task 1.13: Apply one changed square to one compressed FEN rank row."""

    def task_id(self) -> str:
        return "1.13_fen_rank_cell_edit"

    def tier(self) -> int:
        return 1

    def generate(self) -> Iterator[dict]:
        game_positions = list(self.config.get("game_positions", []))
        fallback_positions = list(self.config.get("fen_pool", []))
        target = self.target_volume()
        shuffle = getattr(self.rng, "shuffle", None)
        if callable(shuffle):
            shuffle(game_positions)
            shuffle(fallback_positions)
        candidate_positions = game_positions + fallback_positions
        count = 0

        for pos in candidate_positions:
            if count >= target:
                return
            raw_start = self.source_row(pos)
            if self.is_blocked(raw_start):
                continue
            board = board_from_raw(raw_start)
            if board is None:
                continue
            legal = list(board.legal_moves)
            if not legal:
                continue

            move = self.rng.choice(legal)
            before = board.copy(stack=False)
            board.push(move)
            detail = _state_tracking_move_detail(before, move, board, 1)
            changed_squares = [
                item
                for item in detail.get("changed_squares", [])
                if isinstance(item, dict)
            ]
            if not changed_squares:
                continue
            changed = changed_squares[count % len(changed_squares)]
            square = chess.parse_square(str(changed["square"]))
            file_name = chess.FILE_NAMES[chess.square_file(square)]
            rank = str(chess.square_rank(square) + 1)
            before_row = _fen_rank_row(before, int(rank))
            # Read the taught row from the actual pushed board so multi-edit
            # ranks (castling, en passant, promotions) stay truthful.
            after_row = _fen_rank_row(board, int(rank))
            answer = _format_rank_cell_edit_answer(rank, before_row, after_row)
            metadata = dict(raw_start.get("metadata", {}))
            metadata.update(
                {
                    "move": move.uci(),
                    "move_kind": str(detail["move_kind"]),
                    "square": str(changed["square"]),
                    "rank": rank,
                    "file": file_name,
                    "before_row": before_row,
                    "after_row": after_row,
                    "before_fen": str(changed["before_fen"]),
                    "after_fen": str(changed["after_fen"]),
                    "expected_answer": answer,
                    "query_kind": "fen_rank_cell_edit",
                }
            )
            raw = {
                **raw_start,
                "move": move.uci(),
                "square": str(changed["square"]),
                "rank": rank,
                "file": file_name,
                "before_row": before_row,
                "after_row": after_row,
                "before_fen": str(changed["before_fen"]),
                "after_fen": str(changed["after_fen"]),
                "metadata": metadata,
            }
            user_text = self.render_template(raw)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class FENBoardEdit(TaskGenerator):
    """Task 1.14: Apply changed square edits to the board-FEN field only."""

    def task_id(self) -> str:
        return "1.14_fen_board_edit"

    def tier(self) -> int:
        return 1

    def generate(self) -> Iterator[dict]:
        game_positions = list(self.config.get("game_positions", []))
        fallback_positions = list(self.config.get("fen_pool", []))
        target = self.target_volume()
        shuffle = getattr(self.rng, "shuffle", None)
        if callable(shuffle):
            shuffle(game_positions)
            shuffle(fallback_positions)
        candidate_positions = game_positions + fallback_positions
        count = 0

        for pos in candidate_positions:
            if count >= target:
                return
            raw_start = self.source_row(pos)
            if self.is_blocked(raw_start):
                continue
            board = board_from_raw(raw_start)
            if board is None:
                continue
            legal = list(board.legal_moves)
            if not legal:
                continue

            move = self.rng.choice(legal)
            before = board.copy(stack=False)
            board_fen_before = before.board_fen()
            board.push(move)
            detail = _state_tracking_move_detail(before, move, board, 1)
            changed_squares = [
                item
                for item in detail.get("changed_squares", [])
                if isinstance(item, dict)
            ]
            if not changed_squares:
                continue
            board_fen_after = board.board_fen()
            edit_text = _format_board_edit_text(changed_squares)
            answer = _format_fen_board_edit_answer(board_fen_after)
            metadata = dict(raw_start.get("metadata", {}))
            metadata.update(
                {
                    "move": move.uci(),
                    "move_kind": str(detail["move_kind"]),
                    "board_fen_before": board_fen_before,
                    "board_fen_after": board_fen_after,
                    "changed_squares": changed_squares,
                    "edit_text": edit_text,
                    "expected_answer": answer,
                    "query_kind": "fen_board_edit",
                }
            )
            raw = {
                **raw_start,
                "move": move.uci(),
                "board_fen_before": board_fen_before,
                "board_fen_after": board_fen_after,
                "edit_text": edit_text,
                "metadata": metadata,
            }
            user_text = self.render_template(raw)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class MoveSquareEdits(TaskGenerator):
    """Task 1.8: List square lookups and rank edits for one legal move."""

    def task_id(self) -> str:
        return "1.8_move_square_edits"

    def tier(self) -> int:
        return 1

    def generate(self) -> Iterator[dict]:
        game_positions = list(self.config.get("game_positions", []))
        fallback_positions = list(self.config.get("fen_pool", []))
        target = self.target_volume()
        shuffle = getattr(self.rng, "shuffle", None)
        if callable(shuffle):
            shuffle(game_positions)
            shuffle(fallback_positions)
        candidate_positions = game_positions + fallback_positions
        count = 0

        for pos in candidate_positions:
            if count >= target:
                return
            raw_start = self.source_row(pos)
            if self.is_blocked(raw_start):
                continue
            board = board_from_raw(raw_start)
            if board is None:
                continue
            legal = list(board.legal_moves)
            if not legal:
                continue

            move = self.rng.choice(legal)
            before = board.copy(stack=False)
            board.push(move)
            detail = _state_tracking_move_detail(before, move, board, 1)
            answer = _format_move_square_edits_answer(detail)
            metadata = dict(raw_start.get("metadata", {}))
            metadata.update(
                {
                    "move": move.uci(),
                    "move_kind": str(detail["move_kind"]),
                    "move_details": detail,
                    "expected_answer": answer,
                    "query_kind": "move_square_edits",
                }
            )
            raw = {
                **raw_start,
                "move": move.uci(),
                "metadata": metadata,
            }
            user_text = self.render_template(raw)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class FENAssembly(TaskGenerator):
    """Task 1.9: Assemble the resulting FEN from explicit one-move edits."""

    def task_id(self) -> str:
        return "1.9_fen_assembly"

    def tier(self) -> int:
        return 1

    def generate(self) -> Iterator[dict]:
        game_positions = list(self.config.get("game_positions", []))
        fallback_positions = list(self.config.get("fen_pool", []))
        target = self.target_volume()
        shuffle = getattr(self.rng, "shuffle", None)
        if callable(shuffle):
            shuffle(game_positions)
            shuffle(fallback_positions)
        candidate_positions = game_positions + fallback_positions
        count = 0

        for pos in candidate_positions:
            if count >= target:
                return
            raw_start = self.source_row(pos)
            if self.is_blocked(raw_start):
                continue
            board = board_from_raw(raw_start)
            if board is None:
                continue
            legal = list(board.legal_moves)
            if not legal:
                continue

            move = self.rng.choice(legal)
            before = board.copy(stack=False)
            start_fen = before.fen()
            board.push(move)
            detail = _state_tracking_move_detail(before, move, board, 1)
            result_fen = board.fen()
            answer = _format_state_tracking_answer([detail], result_fen)
            move_uci = move.uci()

            metadata = dict(raw_start.get("metadata", {}))
            metadata.update(
                {
                    "result_fen": result_fen,
                    "move": move_uci,
                    "moves": move_uci,
                    "n_moves": 1,
                    "move_kind": str(detail["move_kind"]),
                    "move_details": [detail],
                    "expected_answer": answer,
                    "query_kind": "fen_assembly",
                }
            )
            raw = {
                **raw_start,
                "fen": start_fen,
                "move": move_uci,
                "moves": move_uci,
                "n_moves": "1",
                "metadata": metadata,
            }
            user_text = self.render_template(raw)
            user_text = _append_full_fen_state(
                user_text,
                self.build_template_context(raw),
            )
            example = self.format_example(
                raw,
                template_text=user_text,
                assistant_content=answer,
            )
            if self.example_is_blocked(example):
                continue
            yield example
            count += 1


class FENRowApplication(TaskGenerator):
    """Task 1.10: Apply affected compressed FEN rank-row rewrites."""

    def task_id(self) -> str:
        return "1.10_fen_row_application"

    def tier(self) -> int:
        return 1

    def generate(self) -> Iterator[dict]:
        game_positions = list(self.config.get("game_positions", []))
        fallback_positions = list(self.config.get("fen_pool", []))
        target = self.target_volume()
        shuffle = getattr(self.rng, "shuffle", None)
        if callable(shuffle):
            shuffle(game_positions)
            shuffle(fallback_positions)
        candidate_positions = game_positions + fallback_positions
        count = 0

        for pos in candidate_positions:
            if count >= target:
                return
            raw_start = self.source_row(pos)
            if self.is_blocked(raw_start):
                continue
            board = board_from_raw(raw_start)
            if board is None:
                continue
            legal = list(board.legal_moves)
            if not legal:
                continue

            move = self.rng.choice(legal)
            before = board.copy(stack=False)
            start_fen = before.fen()
            board.push(move)
            detail = _state_tracking_move_detail(before, move, board, 1)
            result_fen = board.fen()
            answer = _format_fen_row_application_answer(detail, result_fen)
            move_uci = move.uci()

            metadata = dict(raw_start.get("metadata", {}))
            metadata.update(
                {
                    "result_fen": result_fen,
                    "move": move_uci,
                    "moves": move_uci,
                    "n_moves": 1,
                    "move_kind": str(detail["move_kind"]),
                    "move_details": [detail],
                    "rank_updates": detail["rank_updates"],
                    "expected_answer": answer,
                    "query_kind": "fen_row_application",
                }
            )
            raw = {
                **raw_start,
                "fen": start_fen,
                "move": move_uci,
                "moves": move_uci,
                "n_moves": "1",
                "metadata": metadata,
            }
            user_text = self.render_template(raw)
            user_text = _append_full_fen_state(
                user_text,
                self.build_template_context(raw),
            )
            example = self.format_example(
                raw,
                template_text=user_text,
                assistant_content=answer,
            )
            if self.example_is_blocked(example):
                continue
            yield example
            count += 1


class StateTracking(TaskGenerator):
    """Task 1.5: Apply one move by default and track resulting position."""

    ply_min_config_key = "state_tracking_min_plies"
    ply_max_config_key = "state_tracking_max_plies"
    default_min_plies = _DEFAULT_STATE_TRACKING_MIN_PLIES
    default_max_plies = _DEFAULT_STATE_TRACKING_MAX_PLIES

    def task_id(self) -> str:
        return "1.5_state_tracking"

    def tier(self) -> int:
        return 1

    def _ply_bounds(self) -> tuple[int, int]:
        return _state_tracking_ply_bounds(
            self.config,
            min_key=self.ply_min_config_key,
            max_key=self.ply_max_config_key,
            default_min=self.default_min_plies,
            default_max=self.default_max_plies,
        )

    def generate(self) -> Iterator[dict]:
        game_positions = list(self.config.get("game_positions", []))
        fallback_positions = list(self.config.get("fen_pool", []))
        target = self.target_volume()
        shuffle = getattr(self.rng, "shuffle", None)
        if callable(shuffle):
            shuffle(game_positions)
            shuffle(fallback_positions)
        candidate_positions = game_positions + fallback_positions
        count = 0
        min_plies, max_plies = self._ply_bounds()

        for pos in candidate_positions:
            if count >= target:
                return
            raw_start = self.source_row(pos)
            if self.is_blocked(raw_start):
                continue

            board = board_from_raw(raw_start)
            if board is None:
                continue
            start_fen = board.fen()
            n_moves = self.rng.randint(min_plies, max_plies)

            moves_played = []
            move_details = []
            for _ in range(n_moves):
                legal = list(board.legal_moves)
                if not legal:
                    break
                move = self.rng.choice(legal)
                before = board.copy(stack=False)
                moves_played.append(move.uci())
                board.push(move)
                move_details.append(
                    _state_tracking_move_detail(before, move, board, len(moves_played))
                )

            if len(moves_played) < min_plies:
                continue

            result_fen = board.fen()
            moves_str = " ".join(moves_played)
            answer = f"Result FEN: {result_fen}"

            metadata = dict(raw_start.get("metadata", {}))
            metadata.update({
                "result_fen": result_fen,
                "moves": moves_str,
                "n_moves": len(moves_played),
                "move_kinds": [str(detail["move_kind"]) for detail in move_details],
                "move_details": move_details,
                "expected_answer": answer,
            })
            raw = {
                **raw_start,
                "fen": start_fen,
                "moves": moves_str,
                "n_moves": str(len(moves_played)),
                "metadata": metadata,
            }
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            user_text = _append_full_fen_state(
                user_text,
                self.build_template_context(raw),
            )
            example = self.format_example(
                raw, template_text=user_text, assistant_content=answer
            )
            if self.example_is_blocked(example):
                continue
            yield example
            count += 1


class MultiMoveStateTracking(StateTracking):
    """Task 1.19: Apply 2-3 moves and report only the resulting FEN.

    A separate task id (not a 1.5 config knob) so the frozen 1-ply
    state_tracking metric stays comparable across runs.
    """

    ply_min_config_key = "multi_state_tracking_min_plies"
    ply_max_config_key = "multi_state_tracking_max_plies"
    default_min_plies = 2
    default_max_plies = 3

    def task_id(self) -> str:
        return "1.19_multi_move_state_tracking"
