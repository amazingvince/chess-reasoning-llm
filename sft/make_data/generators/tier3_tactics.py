"""Tier 3: Tactics tasks (~260K examples).

3.1 Available captures
3.2 Threats
3.3 Attacked/defended square analysis
3.4 Tactical patterns (from Lichess puzzles by theme)
3.5 Hanging pieces (undefended pieces under attack)
"""

from __future__ import annotations

from typing import Iterator

import chess

from config.templates import select_template
from generators.base import TaskGenerator

_PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}

_PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


class AvailableCaptures(TaskGenerator):
    """Task 3.1: List all capture moves."""

    def task_id(self) -> str:
        return "3.1_available_captures"

    def tier(self) -> int:
        return 3

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
            board = chess.Board(fen)

            captures = sorted(
                m.uci() for m in board.legal_moves if board.is_capture(m)
            )
            if not captures:
                answer = "No captures available."
            else:
                answer = " ".join(captures)

            raw = {"fen": fen, "is_chess960": is_960}
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class Threats(TaskGenerator):
    """Task 3.2: Identify threats to pieces."""

    def task_id(self) -> str:
        return "3.2_threats"

    def tier(self) -> int:
        return 3

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
            board = chess.Board(fen)

            color = board.turn
            color_name = "white" if color == chess.WHITE else "black"
            opp_color = not color

            raw = {"fen": fen, "color": color_name, "is_chess960": is_960}
            tpl = select_template(self.task_id(), self.rng)

            # Determine if the template asks "what is {color} threatening?"
            # (answer = opponent pieces attacked) or "which {color} pieces
            # are under attack?" (answer = same-color pieces being attacked)
            tpl_lower = tpl.lower()
            if "under attack" in tpl_lower:
                # "Which {color} pieces are under attack?" ->
                # list pieces of {color} that the opponent attacks
                under_attack = []
                for sq in chess.SQUARES:
                    piece = board.piece_at(sq)
                    if piece and piece.color == color:
                        if board.is_attacked_by(opp_color, sq):
                            under_attack.append(
                                f"{_PIECE_NAMES[piece.piece_type]} on {chess.square_name(sq)}"
                            )
                if under_attack:
                    answer = f"{color_name.capitalize()} pieces under attack: {', '.join(under_attack)}."
                else:
                    answer = f"No {color_name} pieces are under attack."
            else:
                # "What is {color} threatening?" / generic ->
                # list opponent pieces attacked by {color}
                threatened = []
                for sq in chess.SQUARES:
                    piece = board.piece_at(sq)
                    if piece and piece.color == opp_color:
                        if board.is_attacked_by(color, sq):
                            threatened.append(
                                f"{_PIECE_NAMES[piece.piece_type]} on {chess.square_name(sq)}"
                            )
                if threatened:
                    answer = f"{color_name.capitalize()} threatens: {', '.join(threatened)}."
                else:
                    answer = f"{color_name.capitalize()} has no immediate threats."

            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class AttackedDefended(TaskGenerator):
    """Task 3.3: Attacked/defended square analysis."""

    def task_id(self) -> str:
        return "3.3_attacked_defended"

    def tier(self) -> int:
        return 3

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
            board = chess.Board(fen)

            # Pick a random square that has a piece or is interesting
            sq = self.rng.choice(chess.SQUARES)
            square_name = chess.square_name(sq)
            piece = board.piece_at(sq)
            piece_name = _PIECE_NAMES.get(piece.piece_type, "piece") if piece else "square"

            white_attackers = board.attackers(chess.WHITE, sq)
            black_attackers = board.attackers(chess.BLACK, sq)
            w_count = len(white_attackers)
            b_count = len(black_attackers)

            w_pieces = [
                f"{_PIECE_NAMES[board.piece_at(s).piece_type]} on {chess.square_name(s)}"
                for s in white_attackers if board.piece_at(s)
            ]
            b_pieces = [
                f"{_PIECE_NAMES[board.piece_at(s).piece_type]} on {chess.square_name(s)}"
                for s in black_attackers if board.piece_at(s)
            ]

            parts = []
            if w_count > 0:
                parts.append(f"Attacked by white ({w_count}): {', '.join(w_pieces)}.")
            if b_count > 0:
                parts.append(f"Attacked by black ({b_count}): {', '.join(b_pieces)}.")
            if not parts:
                parts.append(f"Square {square_name} is not attacked by either side.")

            # Defense: if a piece is on the square, note its defenders
            if piece:
                defenders = board.attackers(piece.color, sq)
                if defenders:
                    d_pieces = [
                        f"{_PIECE_NAMES[board.piece_at(s).piece_type]} on {chess.square_name(s)}"
                        for s in defenders if board.piece_at(s)
                    ]
                    color_name = "white" if piece.color == chess.WHITE else "black"
                    parts.append(f"Defended by {color_name} ({len(d_pieces)}): {', '.join(d_pieces)}.")
                else:
                    parts.append(f"The {piece_name} on {square_name} is not defended.")

            answer = " ".join(parts)
            raw = {"fen": fen, "square": square_name, "piece": piece_name,
                   "is_chess960": is_960}
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class TacticalPatterns(TaskGenerator):
    """Task 3.4: Fork, pin, skewer, etc. from Lichess puzzles by theme."""

    def task_id(self) -> str:
        return "3.4_tactical_patterns"

    def tier(self) -> int:
        return 3

    def generate(self) -> Iterator[dict]:
        puzzles = self.config.get("puzzles", [])
        target = self.target_volume()
        count = 0

        for puzzle in puzzles:
            if count >= target:
                return
            fen = puzzle["fen"]
            if self.is_blocked(fen):
                continue

            themes = puzzle.get("themes", [])
            solution = puzzle.get("solution_first_move", "")
            if not solution:
                continue

            theme_str = ", ".join(themes) if themes else "tactical"
            answer = f"The tactic is {theme_str}. Best move: {solution}"

            raw = {"fen": fen, "is_chess960": False,
                   "metadata": {"themes": themes, "source": "lichess_puzzles",
                                "rating": puzzle.get("rating", 0)}}
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class HangingPieces(TaskGenerator):
    """Task 3.5: Undefended pieces under attack."""

    def task_id(self) -> str:
        return "3.5_hanging_pieces"

    def tier(self) -> int:
        return 3

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
            board = chess.Board(fen)

            hanging = []
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece is None or piece.piece_type == chess.KING:
                    continue

                enemy_color = not piece.color
                friendly_color = piece.color

                is_attacked = board.is_attacked_by(enemy_color, sq)
                is_defended = board.is_attacked_by(friendly_color, sq)

                if is_attacked and not is_defended:
                    color_name = "white" if piece.color == chess.WHITE else "black"
                    hanging.append(
                        f"{color_name} {_PIECE_NAMES[piece.piece_type]} on {chess.square_name(sq)}"
                    )

            if hanging:
                answer = f"Hanging pieces: {', '.join(hanging)}."
            else:
                answer = "No hanging pieces — all attacked pieces are defended."

            raw = {"fen": fen, "is_chess960": is_960}
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1
