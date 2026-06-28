"""Tier 2: Rules tasks (~370K examples).

2.1 Legal move generation
2.2 Piece-specific moves
2.3 Move legality check
2.4 Check/checkmate/stalemate detection
2.5 Special rules (castling, en passant, promotion)
"""

from __future__ import annotations

from random import Random
from typing import Iterator

import chess

from config.templates import select_template
from generators.base import TaskGenerator


class LegalMoveGen(TaskGenerator):
    """Task 2.1: List all legal moves for a position."""

    def task_id(self) -> str:
        return "2.1_legal_move_gen"

    def tier(self) -> int:
        return 2

    def generate(self) -> Iterator[dict]:
        pool = self.config.get("fen_pool", [])
        target = self.target_volume()
        count = 0
        for entry in pool:
            if count >= target:
                return
            fen = entry if isinstance(entry, str) else entry["fen"]
            if self.is_blocked(fen):
                continue
            is_960 = entry.get("is_chess960", False) if isinstance(entry, dict) else False

            board = chess.Board(fen, chess960=is_960)
            legal_moves = sorted(m.uci() for m in board.legal_moves)
            if not legal_moves:
                continue

            answer = " ".join(legal_moves)
            raw = {"fen": fen, "is_chess960": is_960}
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
        pool = self.config.get("fen_pool", [])
        target = self.target_volume()
        count = 0

        piece_names = {
            chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
            chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
        }

        for entry in pool:
            if count >= target:
                return
            fen = entry if isinstance(entry, str) else entry["fen"]
            if self.is_blocked(fen):
                continue
            is_960 = entry.get("is_chess960", False) if isinstance(entry, dict) else False
            board = chess.Board(fen, chess960=is_960)

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
            pname = piece_names.get(piece.piece_type, "piece")

            # Get moves from this square
            moves_from_sq = sorted(
                m.uci() for m in board.legal_moves if m.from_square == sq
            )

            if not moves_from_sq:
                continue

            answer = " ".join(moves_from_sq)
            raw = {"fen": fen, "square": square_name, "piece": pname, "is_chess960": is_960}
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class MoveLegalityCheck(TaskGenerator):
    """Task 2.3: Is a specific move legal? 50/50 legal/illegal split."""

    def task_id(self) -> str:
        return "2.3_move_legality_check"

    def tier(self) -> int:
        return 2

    def generate(self) -> Iterator[dict]:
        pool = self.config.get("fen_pool", [])
        target = self.target_volume()
        count = 0

        for entry in pool:
            if count >= target:
                return
            fen = entry if isinstance(entry, str) else entry["fen"]
            if self.is_blocked(fen):
                continue
            is_960 = entry.get("is_chess960", False) if isinstance(entry, dict) else False
            board = chess.Board(fen, chess960=is_960)

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                continue

            # 50/50 legal vs illegal
            if self.rng.random() < 0.5:
                # Legal move
                move = self.rng.choice(legal_moves)
                move_uci = move.uci()
                answer = "Yes, the move is legal."
            else:
                # Generate an illegal move
                move_uci = _random_illegal_move(board, self.rng)
                if move_uci is None:
                    continue
                answer = "No, the move is not legal."

            raw = {"fen": fen, "move": move_uci, "is_chess960": is_960,
                   "metadata": {"tested_move": move_uci}}
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


def _random_illegal_move(board: chess.Board, rng: Random) -> str | None:
    """Generate a random move that looks plausible but is illegal."""
    legal_set = {m.uci() for m in board.legal_moves}
    # Try random from-to combinations
    for _ in range(50):
        from_sq = rng.choice(chess.SQUARES)
        to_sq = rng.choice(chess.SQUARES)
        if from_sq == to_sq:
            continue
        uci = chess.square_name(from_sq) + chess.square_name(to_sq)
        if uci not in legal_set:
            return uci
    return None


class CheckDetection(TaskGenerator):
    """Task 2.4: Detect check, checkmate, stalemate, or normal position."""

    def task_id(self) -> str:
        return "2.4_check_detection"

    def tier(self) -> int:
        return 2

    def generate(self) -> Iterator[dict]:
        pool = self.config.get("fen_pool", [])
        target = self.target_volume()
        count = 0

        for entry in pool:
            if count >= target:
                return
            fen = entry if isinstance(entry, str) else entry["fen"]
            if self.is_blocked(fen):
                continue
            is_960 = entry.get("is_chess960", False) if isinstance(entry, dict) else False
            board = chess.Board(fen, chess960=is_960)

            if board.is_checkmate():
                answer = "Checkmate."
            elif board.is_stalemate():
                answer = "Stalemate."
            elif board.is_check():
                answer = "Check."
            else:
                answer = "Normal position -- no check, checkmate, or stalemate."

            raw = {"fen": fen, "is_chess960": is_960}
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class SpecialRules(TaskGenerator):
    """Task 2.5: Castling (incl. Chess960), en passant, promotion."""

    def task_id(self) -> str:
        return "2.5_special_rules"

    def tier(self) -> int:
        return 2

    def generate(self) -> Iterator[dict]:
        pool = self.config.get("fen_pool", [])
        target = self.target_volume()
        count = 0

        for entry in pool:
            if count >= target:
                return
            fen = entry if isinstance(entry, str) else entry["fen"]
            if self.is_blocked(fen):
                continue
            is_960 = entry.get("is_chess960", False) if isinstance(entry, dict) else False
            board = chess.Board(fen, chess960=is_960)

            special_moves = _identify_special_moves(board)
            if not special_moves:
                continue

            # Pick a square for templates that need {square}
            square = ""
            if special_moves["promotions"]:
                square = special_moves["promotions"][0]

            raw = {"fen": fen, "square": square, "is_chess960": is_960}
            tpl = select_template(self.task_id(), self.rng)
            tpl_lower = tpl.lower()
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
                if special_moves["promotions"]:
                    promo_sqs = " ".join(special_moves["promotions"])
                    answer_parts.append(
                        f"Promotion possible from: {promo_sqs}. "
                        f"Each pawn can promote to queen, rook, bishop, or knight."
                    )
                answer = " ".join(answer_parts) if answer_parts else "No special moves available."

            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


def _identify_special_moves(board: chess.Board) -> dict:
    """Identify castling, en passant, and promotion possibilities."""
    result = {"castling": [], "en_passant": [], "promotions": []}

    # Castling
    if board.has_kingside_castling_rights(board.turn):
        result["castling"].append("kingside")
    if board.has_queenside_castling_rights(board.turn):
        result["castling"].append("queenside")

    # En passant
    if board.ep_square is not None:
        for move in board.legal_moves:
            if move.to_square == board.ep_square and board.is_en_passant(move):
                result["en_passant"].append(move.uci())

    # Promotions
    promo_squares = set()
    for move in board.legal_moves:
        if move.promotion:
            promo_squares.add(chess.square_name(move.from_square))
    result["promotions"] = sorted(promo_squares)

    return result
