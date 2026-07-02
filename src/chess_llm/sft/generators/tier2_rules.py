"""Tier 2: Rules tasks (~410K examples).

2.0 Side-to-move piece inventory
2.1 Legal move generation
2.2 Piece-specific moves
2.3 Move legality check
2.4 Check/checkmate/stalemate detection
2.5 Special rules (castling, en passant, promotion)
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from random import Random
from typing import Iterator

import chess

from chess_llm.core.legality import (
    classify_move_legality,
    format_legality_answer,
    random_illegal_move_with_reason,
)
from chess_llm.sft.context import board_from_raw
from chess_llm.sft.generators.base import TaskGenerator
from chess_llm.sft.templates import TEMPLATES, select_template


_PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def _side_name(color: bool) -> str:
    return "white" if color == chess.WHITE else "black"


def _piece_phrase(piece: chess.Piece) -> str:
    return f"{_side_name(piece.color)} {_PIECE_NAMES[piece.piece_type]}"


def _side_piece_squares(board: chess.Board) -> list[int]:
    return [
        square
        for square in chess.SQUARES
        if (piece := board.piece_at(square)) is not None and piece.color == board.turn
    ]


def _legal_moves_by_piece(board: chess.Board) -> dict[str, list[str]]:
    grouped = {
        chess.square_name(square): []
        for square in _side_piece_squares(board)
    }
    for move in sorted(board.legal_moves, key=lambda item: item.uci()):
        square = chess.square_name(move.from_square)
        grouped.setdefault(square, []).append(move.uci())
    return grouped


def _side_piece_inventory(board: chess.Board) -> list[dict[str, str]]:
    inventory = []
    for square in _side_piece_squares(board):
        piece = board.piece_at(square)
        if piece is None:
            continue
        inventory.append(
            {
                "square": chess.square_name(square),
                "piece": _piece_phrase(piece),
            }
        )
    return inventory


def _format_side_piece_inventory(board: chess.Board) -> tuple[str, list[dict[str, str]]]:
    inventory = _side_piece_inventory(board)
    pieces = "; ".join(
        f"{item['square']} {item['piece']}"
        for item in inventory
    )
    if not pieces:
        pieces = "none"
    answer = f"Side to move: {_side_name(board.turn)}.\nPieces: {pieces}."
    return answer, inventory


def _format_compact_legal_moves(board: chess.Board) -> tuple[str, dict[str, list[str]], str]:
    legal_moves = sorted(move.uci() for move in board.legal_moves)
    grouped = _legal_moves_by_piece(board)
    final_moves = " ".join(legal_moves) if legal_moves else "none"
    answer = f"Side to move: {_side_name(board.turn)}.\nLegal moves: {final_moves}"
    return answer, grouped, final_moves


def _format_grouped_legal_moves(board: chess.Board) -> tuple[str, dict[str, list[str]], str]:
    legal_moves = sorted(move.uci() for move in board.legal_moves)
    grouped = _legal_moves_by_piece(board)
    pieces = []
    move_lines = []
    for square in _side_piece_squares(board):
        square_name = chess.square_name(square)
        piece = board.piece_at(square)
        if piece is None:
            continue
        phrase = _piece_phrase(piece)
        pieces.append(f"{square_name} {phrase}")
        piece_moves = grouped.get(square_name, [])
        move_text = " ".join(piece_moves) if piece_moves else "no legal moves"
        move_lines.append(f"{square_name} {phrase}: {move_text}")

    final_moves = " ".join(legal_moves) if legal_moves else "none"
    answer = (
        f"Side to move: {_side_name(board.turn)}.\n"
        f"Pieces to inspect: {'; '.join(pieces)}.\n"
        "Moves by piece:\n"
        + "\n".join(move_lines)
        + f"\nAll legal moves: {final_moves}"
    )
    return answer, grouped, final_moves


_LEGAL_DECOMPOSITION_FALLBACK_FENS: tuple[str, ...] = (
    "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1",
    "4r2k/8/8/8/8/8/8/R3K3 w - - 0 1",
    "7k/8/8/8/8/8/8/R3K3 w - - 0 1",
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


def _candidate_piece_squares(
    board: chess.Board,
    *,
    prefer_rejected: bool = False,
) -> list[int]:
    candidates: list[tuple[int, list[str], list[str]]] = []
    for square in _side_piece_squares(board):
        pseudo = _pseudo_legal_moves_from_square(board, square)
        if not pseudo:
            continue
        legal = _legal_moves_from_square(board, square)
        candidates.append((square, pseudo, legal))
    if prefer_rejected:
        rejected = [
            item
            for item in candidates
            if sorted(set(item[1]) - set(item[2]))
        ]
        if rejected:
            return [item[0] for item in rejected]
    return [item[0] for item in candidates]


def _select_piece_square(
    board: chess.Board,
    *,
    prefer_rejected: bool = False,
) -> int | None:
    candidates = _candidate_piece_squares(board, prefer_rejected=prefer_rejected)
    if not candidates:
        return None
    return candidates[0]


def _format_piece_pseudo_legal_answer(board: chess.Board, square: int) -> str:
    square_name = chess.square_name(square)
    pseudo = _pseudo_legal_moves_from_square(board, square)
    return f"Pseudo-legal moves from {square_name}: {_move_text(pseudo)}"


def _format_piece_legal_filter_answer(board: chess.Board, square: int) -> str:
    square_name = chess.square_name(square)
    pseudo = _pseudo_legal_moves_from_square(board, square)
    legal = _legal_moves_from_square(board, square)
    rejected = sorted(set(pseudo) - set(legal))
    return "\n".join(
        [
            f"Pseudo-legal from {square_name}: {_move_text(pseudo)}.",
            f"Legal: {_move_text(legal)}.",
            f"Rejected: {_rejected_move_text(board, rejected)}.",
        ]
    )


def _select_king_safety_move(board: chess.Board) -> str | None:
    rejected: list[str] = []
    legal: list[str] = []
    for square in _side_piece_squares(board):
        for move_uci in _pseudo_legal_moves_from_square(board, square):
            classification = classify_move_legality(board, move_uci)
            if classification.is_legal:
                legal.append(move_uci)
            else:
                rejected.append(move_uci)
    if rejected:
        return sorted(rejected)[0]
    if legal:
        return sorted(legal)[0]
    return None


def _format_king_safety_filter_answer(board: chess.Board, move_uci: str) -> str:
    classification = classify_move_legality(board, move_uci)
    pseudo = False
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


def _format_legal_moves_by_piece_answer(
    board: chess.Board,
) -> tuple[str, dict[str, list[str]], str]:
    legal_moves = sorted(move.uci() for move in board.legal_moves)
    grouped = _legal_moves_by_piece(board)
    pieces = []
    move_lines = []
    for square in _side_piece_squares(board):
        square_name = chess.square_name(square)
        piece = board.piece_at(square)
        if piece is None:
            continue
        phrase = _piece_phrase(piece)
        pieces.append(f"{square_name} {phrase}")
        piece_moves = grouped.get(square_name, [])
        move_lines.append(f"{square_name} {phrase}: {_move_text(piece_moves, empty='no legal moves')}")

    final_moves = _move_text(legal_moves)
    answer = (
        f"Side to move: {_side_name(board.turn)}.\n"
        f"Pieces: {'; '.join(pieces) if pieces else 'none'}.\n"
        "Moves by piece:\n"
        + "\n".join(move_lines)
        + f"\nAll legal moves: {final_moves}"
    )
    return answer, grouped, final_moves


def _legal_decomposition_pool(config: dict) -> list[dict]:
    pool = list(config.get("fen_pool", []))
    pool.extend({"fen": fen, "source": "synthetic_legal_decomposition"} for fen in _LEGAL_DECOMPOSITION_FALLBACK_FENS)
    return pool


class SidePieceInventory(TaskGenerator):
    """Task 2.0: List side-to-move pieces before move generation."""

    def task_id(self) -> str:
        return "2.0_side_piece_inventory"

    def tier(self) -> int:
        return 2

    def generate(self) -> Iterator[dict]:
        pool = list(self.config.get("fen_pool", []))
        self.rng.shuffle(pool)
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

            answer, inventory = _format_side_piece_inventory(board)
            metadata = dict(raw.get("metadata", {}))
            metadata.update(
                {
                    "side_to_move": _side_name(board.turn),
                    "side_piece_inventory": inventory,
                    "expected_answer": answer,
                }
            )
            raw["metadata"] = metadata
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class LegalMoveGen(TaskGenerator):
    """Task 2.1: List all legal moves for a position."""

    def task_id(self) -> str:
        return "2.1_legal_move_gen"

    def tier(self) -> int:
        return 2

    def generate(self) -> Iterator[dict]:
        pool = list(self.config.get("fen_pool", []))
        self.rng.shuffle(pool)
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
            legal_moves = sorted(m.uci() for m in board.legal_moves)
            if not legal_moves:
                continue

            answer, grouped, final_moves = _format_compact_legal_moves(board)
            metadata = dict(raw.get("metadata", {}))
            metadata.update(
                {
                    "legal_move_count": len(legal_moves),
                    "legal_moves": final_moves,
                    "legal_moves_by_piece": grouped,
                    "side_to_move": "white" if board.turn == chess.WHITE else "black",
                    "in_check": board.is_check(),
                    "expected_answer": answer,
                }
            )
            raw["metadata"] = metadata
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class PieceSpecificMoves(TaskGenerator):
    """Task 2.2: Legal moves for a specific piece/square."""

    def task_id(self) -> str:
        return "2.2_piece_specific_moves"

    def tier(self) -> int:
        return 2

    def generate(self) -> Iterator[dict]:
        pool = list(self.config.get("fen_pool", []))
        self.rng.shuffle(pool)
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

            # Pick a random piece belonging to the side to move
            pieces_of_side = []
            for sq in chess.SQUARES:
                p = board.piece_at(sq)
                if p and p.color == board.turn:
                    pieces_of_side.append(sq)

            if not pieces_of_side:
                continue

            sq = self.rng.choice(pieces_of_side)
            square_name = chess.square_name(sq)
            piece = board.piece_at(sq)
            pname = _PIECE_NAMES.get(piece.piece_type, "piece")

            # Get moves from this square
            moves_from_sq = sorted(
                m.uci() for m in board.legal_moves if m.from_square == sq
            )

            answer = " ".join(moves_from_sq) if moves_from_sq else "No legal moves."
            metadata = dict(raw.get("metadata", {}))
            metadata.update(
                {
                    "source_square": square_name,
                    "piece": pname,
                    "expected_moves": answer,
                    "expected_answer": answer,
                }
            )
            raw.update({"square": square_name, "piece": pname, "metadata": metadata})
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class PiecePseudoLegalMoves(TaskGenerator):
    """Task 2.6: List pseudo-legal moves for one side-to-move piece."""

    def task_id(self) -> str:
        return "2.6_piece_pseudo_legal_moves"

    def tier(self) -> int:
        return 2

    def generate(self) -> Iterator[dict]:
        pool = _legal_decomposition_pool(self.config)
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
            square = _select_piece_square(board)
            if square is None:
                continue
            square_name = chess.square_name(square)
            piece = board.piece_at(square)
            answer = _format_piece_pseudo_legal_answer(board, square)
            metadata = dict(raw.get("metadata", {}))
            metadata.update(
                {
                    "source_square": square_name,
                    "piece": _PIECE_NAMES.get(piece.piece_type, "piece") if piece else "piece",
                    "pseudo_legal_moves": _move_text(_pseudo_legal_moves_from_square(board, square)),
                    "expected_answer": answer,
                }
            )
            raw.update({"square": square_name, "metadata": metadata})
            user_text = self.render_template(raw)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class PieceLegalFilter(TaskGenerator):
    """Task 2.7: Split one piece's pseudo-legal moves into legal/rejected."""

    def task_id(self) -> str:
        return "2.7_piece_legal_filter"

    def tier(self) -> int:
        return 2

    def generate(self) -> Iterator[dict]:
        pool = _legal_decomposition_pool(self.config)
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
            square = _select_piece_square(board, prefer_rejected=True)
            if square is None:
                continue
            square_name = chess.square_name(square)
            pseudo = _pseudo_legal_moves_from_square(board, square)
            legal = _legal_moves_from_square(board, square)
            rejected = sorted(set(pseudo) - set(legal))
            answer = _format_piece_legal_filter_answer(board, square)
            metadata = dict(raw.get("metadata", {}))
            metadata.update(
                {
                    "source_square": square_name,
                    "pseudo_legal_moves": _move_text(pseudo),
                    "legal_moves": _move_text(legal),
                    "rejected_moves": _move_text(rejected),
                    "expected_answer": answer,
                }
            )
            raw.update({"square": square_name, "metadata": metadata})
            user_text = self.render_template(raw)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class KingSafetyFilter(TaskGenerator):
    """Task 2.8: Decide whether one pseudo-legal move leaves the king safe."""

    def task_id(self) -> str:
        return "2.8_king_safety_filter"

    def tier(self) -> int:
        return 2

    def generate(self) -> Iterator[dict]:
        pool = _legal_decomposition_pool(self.config)
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
            move_uci = _select_king_safety_move(board)
            if move_uci is None:
                continue
            classification = classify_move_legality(board, move_uci)
            answer = _format_king_safety_filter_answer(board, move_uci)
            metadata = dict(raw.get("metadata", {}))
            metadata.update(
                {
                    "tested_move": move_uci,
                    "expected_is_legal": classification.is_legal,
                    "legality_reason_label": classification.reason_label,
                    "expected_answer": answer,
                }
            )
            raw.update({"move": move_uci, "metadata": metadata})
            user_text = self.render_template(raw)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class LegalMovesByPiece(TaskGenerator):
    """Task 2.9: Group full legal move generation by side-to-move piece."""

    def task_id(self) -> str:
        return "2.9_legal_moves_by_piece"

    def tier(self) -> int:
        return 2

    def generate(self) -> Iterator[dict]:
        pool = list(self.config.get("fen_pool", []))
        self.rng.shuffle(pool)
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
            legal_moves = sorted(move.uci() for move in board.legal_moves)
            if not legal_moves:
                continue
            answer, grouped, final_moves = _format_legal_moves_by_piece_answer(board)
            metadata = dict(raw.get("metadata", {}))
            metadata.update(
                {
                    "legal_move_count": len(legal_moves),
                    "legal_moves": final_moves,
                    "legal_moves_by_piece": grouped,
                    "side_to_move": _side_name(board.turn),
                    "expected_answer": answer,
                }
            )
            raw["metadata"] = metadata
            user_text = self.render_template(raw)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class MoveLegalityCheck(TaskGenerator):
    """Task 2.3: Is a specific move legal? 50/50 legal/illegal split."""

    def task_id(self) -> str:
        return "2.3_move_legality_check"

    def tier(self) -> int:
        return 2

    def generate(self) -> Iterator[dict]:
        pool = list(self.config.get("fen_pool", []))
        self.rng.shuffle(pool)
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

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                continue

            # 50/50 legal vs illegal
            if self.rng.random() < 0.5:
                # Legal move
                move = self.rng.choice(legal_moves)
                move_uci = move.uci()
                classification = classify_move_legality(board, move_uci)
                answer = format_legality_answer(
                    classification.is_legal,
                    classification.reason_label,
                )
                expected_is_legal = True
                reason_label = classification.reason_label
            else:
                # Generate an illegal move
                illegal = random_illegal_move_with_reason(board, self.rng)
                if illegal is None:
                    continue
                move_uci, reason_label = illegal
                answer = format_legality_answer(False, reason_label)
                expected_is_legal = False

            metadata = dict(raw.get("metadata", {}))
            metadata.update(
                {
                    "tested_move": move_uci,
                    "expected_is_legal": expected_is_legal,
                    "legality_reason_label": reason_label,
                    "expected_answer": answer,
                }
            )
            raw.update({"move": move_uci, "metadata": metadata})
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


def _random_illegal_move(board: chess.Board, rng: Random) -> str | None:
    """Generate a random move that looks plausible but is illegal."""
    illegal = random_illegal_move_with_reason(board, rng)
    if illegal is None:
        return None
    return illegal[0]


class CheckDetection(TaskGenerator):
    """Task 2.4: Detect check, checkmate, stalemate, or normal position."""

    def task_id(self) -> str:
        return "2.4_check_detection"

    def tier(self) -> int:
        return 2

    def generate(self) -> Iterator[dict]:
        pool = self.config.get("fen_pool", [])
        target = self.target_volume()
        buckets: dict[str, list[dict]] = {
            "normal": [],
            "check": [],
            "checkmate": [],
            "stalemate": [],
        }

        for entry in pool:
            raw = self.source_row(entry)
            if self.is_blocked(raw):
                continue
            board = board_from_raw(raw)
            if board is None:
                continue
            label, _answer = _check_state_answer(board)
            buckets[label].append(raw)

        source_rows = [
            (label, raw)
            for label, rows in buckets.items()
            for raw in rows
        ]
        if target == 1 and source_rows:
            selected_rows = [source_rows[0]]
        else:
            synthetic_target = max(1, target)
            for label, fen in _synthetic_check_state_fens(synthetic_target):
                raw = {"fen": fen, "source": "synthetic_check_detection"}
                if self.is_blocked(raw):
                    continue
                buckets[label].append(raw)

            quotas = _bucket_quotas(
                target,
                [
                    ("normal", 0.55),
                    ("check", 0.25),
                    ("checkmate", 0.10),
                    ("stalemate", 0.10),
                ],
            )
            selected_rows = _select_bucketed_rows(buckets, quotas, target, self.rng)

        for _bucket, raw in selected_rows:
            board = board_from_raw(raw)
            if board is None:
                continue
            label, answer = _check_state_answer(board)
            metadata = dict(raw.get("metadata", {}))
            metadata.update({"state_label": label, "expected_answer": answer})
            raw["metadata"] = metadata
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)


class SpecialRules(TaskGenerator):
    """Task 2.5: Castling (incl. Chess960), en passant, promotion."""

    def task_id(self) -> str:
        return "2.5_special_rules"

    def tier(self) -> int:
        return 2

    def generate(self) -> Iterator[dict]:
        pool = self.config.get("fen_pool", [])
        target = self.target_volume()
        buckets: dict[str, list[dict]] = {
            "castling": [],
            "en_passant": [],
            "promotion": [],
            "none": [],
        }

        for entry in pool:
            raw = self.source_row(entry)
            if self.is_blocked(raw):
                continue
            board = board_from_raw(raw)
            if board is None:
                continue
            special_moves = _identify_special_moves(board)
            bucket = _special_rule_bucket(special_moves)
            buckets[bucket].append(raw)

        source_rows = [
            (bucket, raw)
            for bucket, rows in buckets.items()
            for raw in rows
        ]
        if target == 1 and source_rows:
            selected_rows = [source_rows[0]]
        else:
            synthetic_target = max(1, target)
            for _bucket, fen in _synthetic_special_rule_fens(synthetic_target):
                raw = {"fen": fen, "source": "synthetic_special_rules"}
                if self.is_blocked(raw):
                    continue
                board = board_from_raw(raw)
                if board is None:
                    continue
                special_moves = _identify_special_moves(board)
                bucket = _special_rule_bucket(special_moves)
                buckets[bucket].append(raw)

            quotas = _bucket_quotas(
                target,
                [
                    ("none", 0.50),
                    ("castling", 0.20),
                    ("en_passant", 0.15),
                    ("promotion", 0.15),
                ],
            )
            selected_rows = _select_bucketed_rows(buckets, quotas, target, self.rng)

        for bucket, raw in selected_rows:
            board = board_from_raw(raw)
            if board is None:
                continue
            special_moves = _identify_special_moves(board)
            bucket = _special_rule_bucket(special_moves)

            # Pick a square for templates that need {square}
            square = ""
            if special_moves["promotions"]:
                square = special_moves["promotions"][0]

            metadata = dict(raw.get("metadata", {}))
            raw["square"] = square
            tpl = _select_special_template(self.rng, bucket)
            tpl_lower = tpl.lower()
            if "promotion options" in tpl_lower and not square:
                continue
            is_generic_special_prompt = (
                "special moves" in tpl_lower
                or "special rules" in tpl_lower
                or "identify any special" in tpl_lower
            )

            if "en passant" in tpl_lower and not is_generic_special_prompt:
                if special_moves["en_passant"]:
                    answer = f"En passant possible: {' '.join(special_moves['en_passant'])}."
                else:
                    answer = "No en passant is possible in this position."
            elif ("castle" in tpl_lower or "castling" in tpl_lower) and not is_generic_special_prompt:
                if special_moves["castling"]:
                    answer = f"Castling available: {', '.join(special_moves['castling'])}."
                else:
                    answer = "No castling is available for the side to move."
            elif "promotion options" in tpl_lower:
                # List specific promotion moves for this pawn
                promo_moves = []
                if square:
                    for move in board.legal_moves:
                        if move.promotion and chess.square_name(move.from_square) == square:
                            promo_names = {chess.QUEEN: "queen", chess.ROOK: "rook",
                                           chess.BISHOP: "bishop", chess.KNIGHT: "knight"}
                            dest = chess.square_name(move.to_square)
                            pname = promo_names.get(move.promotion, "queen")
                            promo_moves.append(f"{move.uci()} (promote to {pname} on {dest})")
                if promo_moves:
                    answer = (
                        f"The pawn on {square} can promote with: "
                        + ", ".join(promo_moves) + "."
                    )
                else:
                    answer = "No promotion is available for the side to move."
            else:
                answer_parts = []
                if special_moves["castling"]:
                    sides = ", ".join(special_moves["castling"])
                    answer_parts.append(f"Castling available: {sides}.")
                if special_moves["en_passant"]:
                    ep_moves = " ".join(special_moves["en_passant"])
                    answer_parts.append(f"En passant possible: {ep_moves}.")
                if special_moves["promotion_moves"]:
                    promotion_moves = " ".join(special_moves["promotion_moves"])
                    answer_parts.append(f"Promotion possible: {promotion_moves}.")
                answer = " ".join(answer_parts) if answer_parts else "No special moves available."

            metadata.update(
                {
                    "expected_answer": answer,
                    "special_rule_bucket": bucket,
                    "castling": " ".join(special_moves["castling"]),
                    "en_passant": " ".join(special_moves["en_passant"]),
                    "promotions": " ".join(special_moves["promotions"]),
                    "promotion_moves": " ".join(special_moves["promotion_moves"]),
                }
            )
            raw["metadata"] = metadata
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)


def _identify_special_moves(board: chess.Board) -> dict:
    """Identify castling, en passant, and promotion possibilities."""
    result = {
        "castling": [],
        "en_passant": [],
        "promotions": [],
        "promotion_moves": [],
    }

    # Castling rights are only permission to castle later; availability
    # requires a legal castling move in the current position.
    result["castling"] = _legal_castling_sides(board)

    # En passant
    if board.ep_square is not None:
        for move in board.legal_moves:
            if move.to_square == board.ep_square and board.is_en_passant(move):
                result["en_passant"].append(move.uci())

    # Promotions
    promo_squares = set()
    promo_moves = []
    for move in board.legal_moves:
        if move.promotion:
            promo_squares.add(chess.square_name(move.from_square))
            promo_moves.append(move.uci())
    result["promotions"] = sorted(promo_squares)
    result["promotion_moves"] = sorted(promo_moves)

    return result


def _legal_castling_sides(board: chess.Board) -> list[str]:
    """Return legal castling sides for the side to move, in stable order."""
    sides: set[str] = set()
    for move in board.legal_moves:
        if not board.is_castling(move):
            continue
        if board.is_kingside_castling(move):
            sides.add("kingside")
        elif board.is_queenside_castling(move):
            sides.add("queenside")

    return [side for side in ("kingside", "queenside") if side in sides]


def _check_state_answer(board: chess.Board) -> tuple[str, str]:
    if board.is_checkmate():
        return "checkmate", "Checkmate."
    if board.is_stalemate():
        return "stalemate", "Stalemate."
    if board.is_check():
        return "check", "Check."
    return "normal", "Normal position -- no check, checkmate, or stalemate."


def _bucket_quotas(target: int, weights: list[tuple[str, float]]) -> dict[str, int]:
    if target <= 0:
        return {bucket: 0 for bucket, _weight in weights}

    total_weight = sum(weight for _bucket, weight in weights)
    raw_counts = [
        (bucket, target * weight / total_weight)
        for bucket, weight in weights
    ]
    quotas = {bucket: int(count) for bucket, count in raw_counts}
    remaining = target - sum(quotas.values())
    remainders = sorted(
        raw_counts,
        key=lambda item: item[1] - int(item[1]),
        reverse=True,
    )
    for bucket, _count in remainders[:remaining]:
        quotas[bucket] += 1
    return quotas


def _select_bucketed_rows(
    buckets: dict[str, list[dict]],
    quotas: dict[str, int],
    target: int,
    rng: Random,
) -> list[tuple[str, dict]]:
    selected: list[tuple[str, dict]] = []
    shuffled: dict[str, list[dict]] = {}
    for bucket, rows in buckets.items():
        shuffled[bucket] = list(rows)
        rng.shuffle(shuffled[bucket])

    bucket_offsets: Counter[str] = Counter()
    for bucket, quota in quotas.items():
        rows = shuffled.get(bucket, [])
        if not rows:
            continue
        for _ in range(quota):
            selected.append((bucket, dict(rows[bucket_offsets[bucket] % len(rows)])))
            bucket_offsets[bucket] += 1

    available = [bucket for bucket, rows in shuffled.items() if rows]
    available_idx = 0
    while len(selected) < target and available:
        bucket = available[available_idx % len(available)]
        rows = shuffled[bucket]
        selected.append((bucket, dict(rows[bucket_offsets[bucket] % len(rows)])))
        bucket_offsets[bucket] += 1
        available_idx += 1

    rng.shuffle(selected)
    return selected[:target]


def _special_rule_bucket(special_moves: dict) -> str:
    if special_moves.get("en_passant"):
        return "en_passant"
    if special_moves.get("promotion_moves") or special_moves.get("promotions"):
        return "promotion"
    if special_moves.get("castling"):
        return "castling"
    return "none"


def _select_special_template(rng: Random, bucket: str) -> str:
    template = select_template("2.5_special_rules", rng)
    if getattr(select_template, "__module__", "") != "chess_llm.sft.templates":
        return template
    if _special_template_matches_bucket(template, bucket):
        return template

    candidates = [
        candidate
        for candidate in TEMPLATES["2.5_special_rules"]
        if _special_template_matches_bucket(candidate, bucket)
    ]
    if not candidates:
        return template
    return rng.choice(candidates)


def _special_template_matches_bucket(template: str, bucket: str) -> bool:
    lower = template.lower()
    is_generic = (
        "special moves" in lower
        or "special rules" in lower
        or "identify any special" in lower
    )
    if bucket == "castling":
        return is_generic or "castle" in lower or "castling" in lower
    if bucket == "en_passant":
        return is_generic or "en passant" in lower
    if bucket == "promotion":
        return is_generic or "promotion options" in lower
    if bucket == "none":
        return "promotion options" not in lower
    return True


@lru_cache(maxsize=8)
def _synthetic_check_state_fens(max_per_label: int) -> tuple[tuple[str, str], ...]:
    max_per_label = max(1, max_per_label)
    rows: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()

    for attacking_color, defending_color in (
        (chess.WHITE, chess.BLACK),
        (chess.BLACK, chess.WHITE),
    ):
        for defending_king in chess.SQUARES:
            for attacking_king in chess.SQUARES:
                if attacking_king == defending_king:
                    continue
                for queen_square in chess.SQUARES:
                    if queen_square in (defending_king, attacking_king):
                        continue
                    board = chess.Board.empty()
                    board.set_piece_at(
                        defending_king,
                        chess.Piece(chess.KING, defending_color),
                    )
                    board.set_piece_at(
                        attacking_king,
                        chess.Piece(chess.KING, attacking_color),
                    )
                    board.set_piece_at(
                        queen_square,
                        chess.Piece(chess.QUEEN, attacking_color),
                    )
                    board.turn = defending_color
                    board.castling_rights = 0
                    board.ep_square = None
                    if not board.is_valid():
                        continue
                    label, _answer = _check_state_answer(board)
                    if label == "normal" or counts[label] >= max_per_label:
                        continue
                    fen = board.fen()
                    if fen in seen:
                        continue
                    seen.add(fen)
                    rows.append((label, fen))
                    counts[label] += 1
                    if all(
                        counts[label_name] >= max_per_label
                        for label_name in ("check", "checkmate", "stalemate")
                    ):
                        return tuple(rows)

    return tuple(rows)


@lru_cache(maxsize=8)
def _synthetic_special_rule_fens(max_per_bucket: int) -> tuple[tuple[str, str], ...]:
    max_per_bucket = max(1, max_per_bucket)
    rows: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()

    def add(bucket: str, board: chess.Board) -> None:
        if counts[bucket] >= max_per_bucket or not board.is_valid():
            return
        actual_bucket = _special_rule_bucket(_identify_special_moves(board))
        if actual_bucket != bucket:
            return
        fen = board.fen()
        if fen in seen:
            return
        seen.add(fen)
        rows.append((bucket, fen))
        counts[bucket] += 1

    _add_synthetic_castling_rows(add, max_per_bucket, counts)
    _add_synthetic_en_passant_rows(add, max_per_bucket, counts)
    _add_synthetic_promotion_rows(add, max_per_bucket, counts)
    _add_synthetic_no_special_rows(add, max_per_bucket, counts)
    return tuple(rows)


def _add_synthetic_castling_rows(add, max_per_bucket: int, counts: Counter[str]) -> None:
    occupied = {chess.A1, chess.E1, chess.H1, chess.A8, chess.E8, chess.H8}
    filler_squares = [
        sq
        for sq in chess.SQUARES
        if chess.square_rank(sq) in range(1, 6) and sq not in occupied
    ]
    right_sets = [
        chess.BB_A1 | chess.BB_H1 | chess.BB_A8 | chess.BB_H8,
        chess.BB_A1 | chess.BB_H1,
        chess.BB_A8 | chess.BB_H8,
        chess.BB_H1 | chess.BB_H8,
        chess.BB_A1 | chess.BB_A8,
    ]

    for turn in (chess.WHITE, chess.BLACK):
        for rights in right_sets:
            for first_extra in [None, *filler_squares]:
                for second_extra in [None, *filler_squares]:
                    if counts["castling"] >= max_per_bucket:
                        return
                    if first_extra is not None and first_extra == second_extra:
                        continue
                    board = chess.Board.empty()
                    for sq, color in (
                        (chess.E1, chess.WHITE),
                        (chess.A1, chess.WHITE),
                        (chess.H1, chess.WHITE),
                        (chess.E8, chess.BLACK),
                        (chess.A8, chess.BLACK),
                        (chess.H8, chess.BLACK),
                    ):
                        piece_type = chess.KING if sq in (chess.E1, chess.E8) else chess.ROOK
                        board.set_piece_at(sq, chess.Piece(piece_type, color))
                    if first_extra is not None:
                        board.set_piece_at(first_extra, chess.Piece(chess.PAWN, chess.WHITE))
                    if second_extra is not None:
                        board.set_piece_at(second_extra, chess.Piece(chess.PAWN, chess.BLACK))
                    board.turn = turn
                    board.castling_rights = rights
                    board.ep_square = None
                    add("castling", board)


def _add_synthetic_en_passant_rows(add, max_per_bucket: int, counts: Counter[str]) -> None:
    king_pairs = [
        (chess.E1, chess.E8),
        (chess.A1, chess.H8),
        (chess.H1, chess.A8),
        (chess.C1, chess.F8),
        (chess.F1, chess.C8),
    ]
    for white_king, black_king in king_pairs:
        for pawn_file in range(8):
            for captured_file in (pawn_file - 1, pawn_file + 1):
                if counts["en_passant"] >= max_per_bucket:
                    return
                if not 0 <= captured_file < 8:
                    continue

                board = chess.Board.empty()
                board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
                board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
                board.set_piece_at(chess.square(pawn_file, 4), chess.Piece(chess.PAWN, chess.WHITE))
                board.set_piece_at(chess.square(captured_file, 4), chess.Piece(chess.PAWN, chess.BLACK))
                board.turn = chess.WHITE
                board.castling_rights = 0
                board.ep_square = chess.square(captured_file, 5)
                add("en_passant", board)

                board = chess.Board.empty()
                board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
                board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
                board.set_piece_at(chess.square(pawn_file, 3), chess.Piece(chess.PAWN, chess.BLACK))
                board.set_piece_at(chess.square(captured_file, 3), chess.Piece(chess.PAWN, chess.WHITE))
                board.turn = chess.BLACK
                board.castling_rights = 0
                board.ep_square = chess.square(captured_file, 2)
                add("en_passant", board)


def _add_synthetic_promotion_rows(add, max_per_bucket: int, counts: Counter[str]) -> None:
    king_pairs = [
        (chess.E1, chess.H8),
        (chess.A1, chess.H8),
        (chess.H1, chess.A8),
        (chess.C1, chess.F8),
        (chess.F1, chess.C8),
    ]
    for white_king, black_king in king_pairs:
        for file_idx in range(8):
            if counts["promotion"] >= max_per_bucket:
                return
            board = chess.Board.empty()
            board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(chess.square(file_idx, 6), chess.Piece(chess.PAWN, chess.WHITE))
            board.turn = chess.WHITE
            board.castling_rights = 0
            board.ep_square = None
            add("promotion", board)

            board = chess.Board.empty()
            board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(chess.square(file_idx, 1), chess.Piece(chess.PAWN, chess.BLACK))
            board.turn = chess.BLACK
            board.castling_rights = 0
            board.ep_square = None
            add("promotion", board)


def _add_synthetic_no_special_rows(add, max_per_bucket: int, counts: Counter[str]) -> None:
    for white_king in chess.SQUARES:
        for black_king in chess.SQUARES:
            if counts["none"] >= max_per_bucket:
                return
            if white_king == black_king:
                continue
            board = chess.Board.empty()
            board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
            board.turn = chess.WHITE
            board.castling_rights = 0
            board.ep_square = None
            add("none", board)
