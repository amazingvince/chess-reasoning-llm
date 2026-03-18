"""Tier 6: Endgame tasks (~140K examples).

6.1 Endgame classification (categorize by material)
6.2 Endgame WDL (Syzygy-backed win/draw/loss evaluation)
6.3 Endgame best move (DTZ-optimal move from tablebases)
6.4 Endgame principles (opposition, Lucena, Philidor, zugzwang, fortress)
"""

from __future__ import annotations

from typing import Iterator

import chess

from config.templates import select_template
from generators.base import TaskGenerator

# WDL labels
_WDL_LABELS = {
    2: "Win for the side to move.",
    1: "Cursed win (win but 50-move rule may prevent it).",
    0: "Draw with best play.",
    -1: "Blessed loss (loss but 50-move rule saves it).",
    -2: "Loss for the side to move.",
}

# Endgame principle keywords mapped to material configs
_ENDGAME_PRINCIPLES: dict[str, list[str]] = {
    "KPK": [
        "Key squares and opposition are critical.",
        "The rule of the square determines if the king can catch the pawn.",
        "With the king in front of the pawn, the stronger side usually wins.",
    ],
    "KRK": [
        "Cut off the defending king with the rook, then use your king to force mate.",
        "Systematic box method: shrink the defending king's space.",
    ],
    "KQKP": [
        "Queen vs pawn on the 7th is tricky — bishop/center pawn may draw.",
        "Use the queen to force the king in front of the pawn, then bring your king closer.",
    ],
    "KRPKR": [
        "Lucena position: bridge-building technique wins when the pawn is on the 7th.",
        "Philidor position: passive defense with the rook on the 6th rank draws.",
        "Cut-off technique: rook cuts off the defending king by files.",
    ],
    "KBBK": [
        "Two bishops can force mate — drive the king to the corner.",
        "Diagonal opposition with bishops controls key squares.",
    ],
    "KBNK": [
        "Bishop and knight mate requires driving the king to the corner matching the bishop's color.",
        "The W-maneuver is the standard technique.",
    ],
}


def _material_signature(board: chess.Board) -> str:
    """Return material signature like 'KRK' or 'KRPKR'."""
    piece_chars = {
        chess.KING: "K", chess.QUEEN: "Q", chess.ROOK: "R",
        chess.BISHOP: "B", chess.KNIGHT: "N", chess.PAWN: "P",
    }
    order = "KQRBNP"
    white = []
    black = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None:
            continue
        ch = piece_chars.get(p.piece_type, "")
        if p.color == chess.WHITE:
            white.append(ch)
        else:
            black.append(ch)
    white.sort(key=lambda c: order.index(c) if c in order else 99)
    black.sort(key=lambda c: order.index(c) if c in order else 99)
    return "".join(white) + "".join(black)


class EndgameClassification(TaskGenerator):
    """Task 6.1: Categorize endgame by material."""

    def task_id(self) -> str:
        return "6.1_endgame_classification"

    def tier(self) -> int:
        return 6

    def generate(self) -> Iterator[dict]:
        endgames = self.config.get("endgame_positions", [])
        target = self.target_volume()
        count = 0
        for eg in endgames:
            if count >= target:
                return
            fen = eg["fen"]
            if self.is_blocked(fen):
                continue

            board = chess.Board(fen)
            sig = eg.get("material", _material_signature(board))

            answer = f"This is a {sig} endgame."

            raw = {"fen": fen, "is_chess960": False,
                   "metadata": {"material": sig, "source": eg.get("source", "syzygy")}}
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class EndgameWDL(TaskGenerator):
    """Task 6.2: Syzygy-backed WDL evaluation."""

    def task_id(self) -> str:
        return "6.2_endgame_wdl"

    def tier(self) -> int:
        return 6

    def generate(self) -> Iterator[dict]:
        endgames = self.config.get("endgame_positions", [])
        target = self.target_volume()
        count = 0
        for eg in endgames:
            if count >= target:
                return
            fen = eg["fen"]
            if self.is_blocked(fen):
                continue

            wdl = eg.get("wdl")
            if wdl is None:
                continue

            answer = _WDL_LABELS.get(wdl, f"WDL value: {wdl}")

            raw = {"fen": fen, "is_chess960": False,
                   "metadata": {"wdl": wdl, "dtz": eg.get("dtz"),
                                "source": "syzygy"}}
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class EndgameBestMove(TaskGenerator):
    """Task 6.3: DTZ-optimal move from tablebases."""

    def task_id(self) -> str:
        return "6.3_endgame_best_move"

    def tier(self) -> int:
        return 6

    def generate(self) -> Iterator[dict]:
        endgames = self.config.get("endgame_positions", [])
        target = self.target_volume()
        count = 0
        for eg in endgames:
            if count >= target:
                return
            fen = eg["fen"]
            if self.is_blocked(fen):
                continue

            best_move = eg.get("best_move", "")
            if not best_move:
                continue

            wdl = eg.get("wdl", 0)
            dtz = eg.get("dtz", 0)
            answer = f"{best_move}"

            raw = {"fen": fen, "is_chess960": False,
                   "metadata": {"best_move": best_move, "wdl": wdl, "dtz": dtz,
                                "source": "syzygy"}}
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class EndgamePrinciples(TaskGenerator):
    """Task 6.4: Opposition, Lucena, Philidor, zugzwang, fortress."""

    def task_id(self) -> str:
        return "6.4_endgame_principles"

    def tier(self) -> int:
        return 6

    def generate(self) -> Iterator[dict]:
        endgames = self.config.get("endgame_positions", [])
        target = self.target_volume()
        count = 0
        for eg in endgames:
            if count >= target:
                return
            fen = eg["fen"]
            if self.is_blocked(fen):
                continue

            board = chess.Board(fen)
            sig = eg.get("material", _material_signature(board))

            # Get principles for this material type
            principles = _ENDGAME_PRINCIPLES.get(sig)
            if not principles:
                # Generic endgame advice
                piece_count = len(board.piece_map())
                if piece_count <= 4:
                    principles = [
                        "With few pieces, king activity is paramount.",
                        "Centralize the king and use it as an attacking piece.",
                    ]
                else:
                    principles = [
                        "Trade down to a winning endgame if you have material advantage.",
                        "Activate your pieces and improve your king position.",
                    ]

            answer = " ".join(self.rng.sample(principles, min(2, len(principles))))

            raw = {"fen": fen, "is_chess960": False,
                   "metadata": {"material": sig, "source": "syzygy"}}
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1
