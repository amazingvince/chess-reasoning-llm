"""Tier 5: Openings tasks (~75K examples).

5.1 Opening identification (name the opening from position/moves)
5.2 Opening continuation (suggest next moves using Polyglot weights)
5.3 Opening principles (plans, character of the position)
"""

from __future__ import annotations

from typing import Iterator

import chess

from config.templates import select_template
from generators.base import TaskGenerator


# Opening character taxonomy for principle descriptions
_OPENING_CHARACTER: dict[str, str] = {
    "A": "Flank opening — flexible, positional, transpositional play.",
    "B": "Semi-open game — asymmetric structures, positional imbalances.",
    "C": "Open game — tactical, piece activity, early castling.",
    "D": "Closed game — strategic, pawn chains, slower development.",
    "E": "Indian defense — hypermodern, fianchettoes, delayed center control.",
}


class OpeningIdentification(TaskGenerator):
    """Task 5.1: Name the opening from position/moves."""

    def task_id(self) -> str:
        return "5.1_opening_identification"

    def tier(self) -> int:
        return 5

    def generate(self) -> Iterator[dict]:
        openings = self.config.get("openings", [])
        target = self.target_volume()
        count = 0

        for opening in openings:
            if count >= target:
                return
            fen = opening["fen"]
            if self.is_blocked(fen):
                continue

            name = opening.get("name", "Unknown")
            eco = opening.get("eco", "")
            uci_moves = opening.get("uci_moves", [])
            moves_str = " ".join(uci_moves) if uci_moves else ""

            answer = f"{name} (ECO: {eco})" if eco else name

            raw = {
                "fen": fen,
                "moves": moves_str,
                "name": name,
                "eco": eco,
                "is_chess960": False,
                "metadata": {"source": "lichess_openings", "eco": eco},
            }
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class OpeningContinuation(TaskGenerator):
    """Task 5.2: Suggest next moves using Polyglot weights."""

    def task_id(self) -> str:
        return "5.2_opening_continuation"

    def tier(self) -> int:
        return 5

    def generate(self) -> Iterator[dict]:
        openings = self.config.get("openings", [])
        book_moves = self.config.get("book_moves", {})  # fen -> [(uci, weight)]
        target = self.target_volume()
        count = 0

        for opening in openings:
            if count >= target:
                return
            fen = opening["fen"]
            if self.is_blocked(fen):
                continue

            name = opening.get("name", "Unknown")
            moves = book_moves.get(fen, [])

            if moves:
                # Format: top moves with weights from Polyglot books
                parts = []
                total_weight = sum(w for _, w in moves)
                for uci, weight in moves[:5]:  # Top 5 moves
                    pct = weight / total_weight * 100 if total_weight > 0 else 0
                    parts.append(f"{uci} ({pct:.0f}%)")
                answer = "Top continuations: " + ", ".join(parts) + "."
            else:
                # No Polyglot data for this position — skip rather than
                # emitting arbitrary legal moves as "continuations."
                continue

            raw = {
                "fen": fen,
                "name": name,
                "is_chess960": False,
                "metadata": {"source": "polyglot_books"},
            }
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class OpeningPrinciples(TaskGenerator):
    """Task 5.3: Plans, character of the position."""

    def task_id(self) -> str:
        return "5.3_opening_principles"

    def tier(self) -> int:
        return 5

    def generate(self) -> Iterator[dict]:
        openings = self.config.get("openings", [])
        target = self.target_volume()
        count = 0

        for opening in openings:
            if count >= target:
                return
            fen = opening["fen"]
            if self.is_blocked(fen):
                continue

            name = opening.get("name", "Unknown")
            eco = opening.get("eco", "")
            eco_volume = opening.get("eco_volume", "")
            if not eco_volume and eco:
                eco_volume = eco[0]  # First letter of ECO code

            character = _OPENING_CHARACTER.get(eco_volume, "")
            board = chess.Board(fen)

            # Generate strategic description based on position features
            desc_parts = [f"This is the {name}."]
            if character:
                desc_parts.append(character)

            # Basic positional observations
            white_developed = sum(
                1 for sq in chess.SQUARES
                if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE
                and board.piece_at(sq).piece_type in (chess.KNIGHT, chess.BISHOP)
                and chess.square_rank(sq) > 0
            )
            black_developed = sum(
                1 for sq in chess.SQUARES
                if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK
                and board.piece_at(sq).piece_type in (chess.KNIGHT, chess.BISHOP)
                and chess.square_rank(sq) < 7
            )

            # Center control
            center = [chess.E4, chess.D4, chess.E5, chess.D5]
            w_center = sum(1 for s in center if board.is_attacked_by(chess.WHITE, s))
            b_center = sum(1 for s in center if board.is_attacked_by(chess.BLACK, s))

            if w_center > b_center:
                desc_parts.append("White has more central control.")
            elif b_center > w_center:
                desc_parts.append("Black has more central control.")

            answer = " ".join(desc_parts)

            raw = {
                "fen": fen,
                "name": name,
                "is_chess960": False,
                "metadata": {"source": "lichess_openings", "eco": eco},
            }
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1
