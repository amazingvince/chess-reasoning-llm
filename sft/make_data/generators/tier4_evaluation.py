"""Tier 4: Evaluation tasks (~150K examples).

4.1 Material balance
4.2 Position evaluation (cp → 5-bucket labels)
4.3 Pawn structure analysis
"""

from __future__ import annotations

from typing import Iterator

import chess

from config.settings import EVAL_BUCKETS
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


def _cp_to_bucket(cp: int) -> str:
    """Map centipawn value to evaluation bucket label."""
    abs_cp = abs(cp)
    for low, high, label in EVAL_BUCKETS:
        if low <= abs_cp < high:
            if cp > 0:
                return f"White has a {label}"
            elif cp < 0:
                return f"Black has a {label}"
            else:
                return label
    return "decisive"


class MaterialBalance(TaskGenerator):
    """Task 4.1: Count material difference."""

    def task_id(self) -> str:
        return "4.1_material_balance"

    def tier(self) -> int:
        return 4

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

            white_mat = 0
            black_mat = 0
            white_pieces: dict[str, int] = {}
            black_pieces: dict[str, int] = {}

            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece is None:
                    continue
                val = _PIECE_VALUES.get(piece.piece_type, 0)
                name = _PIECE_NAMES[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_mat += val
                    white_pieces[name] = white_pieces.get(name, 0) + 1
                else:
                    black_mat += val
                    black_pieces[name] = black_pieces.get(name, 0) + 1

            diff = white_mat - black_mat
            w_str = ", ".join(f"{v} {k}" for k, v in sorted(white_pieces.items()))
            b_str = ", ".join(f"{v} {k}" for k, v in sorted(black_pieces.items()))

            if diff > 0:
                balance = f"White is ahead by {diff} point(s)."
            elif diff < 0:
                balance = f"Black is ahead by {abs(diff)} point(s)."
            else:
                balance = "Material is equal."

            answer = f"White: {w_str} ({white_mat} pts). Black: {b_str} ({black_mat} pts). {balance}"

            raw = {"fen": fen, "is_chess960": is_960}
            tpl = select_template(self.task_id(), self.rng)
            user_text = tpl.format(**raw)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class PositionEvaluation(TaskGenerator):
    """Task 4.2: Map cp → 5-bucket labels. Primary consumer of Position Evals."""

    def task_id(self) -> str:
        return "4.2_position_evaluation"

    def tier(self) -> int:
        return 4

    def generate(self) -> Iterator[dict]:
        evals = self.config.get("position_evals", [])
        target = self.target_volume()
        count = 0
        for ev in evals:
            if count >= target:
                return
            fen = ev["fen"]
            if self.is_blocked(fen):
                continue

            cp = ev.get("cp")
            mate = ev.get("mate")

            if mate is not None:
                if mate > 0:
                    answer = "White has a decisive advantage (forced mate)."
                else:
                    answer = "Black has a decisive advantage (forced mate)."
            elif cp is not None:
                answer = _cp_to_bucket(cp) + "."
            else:
                continue

            raw = {
                "fen": fen,
                "is_chess960": False,
                "metadata": {
                    "source": "lichess_evals",
                    "stockfish_eval_cp": cp,
                    "stockfish_eval_mate": mate,
                },
            }
            tpl = select_template(self.task_id(), self.rng)
            user_text = tpl.format(**raw)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class PawnStructure(TaskGenerator):
    """Task 4.3: Doubled/isolated/passed pawns, pawn chains."""

    def task_id(self) -> str:
        return "4.3_pawn_structure"

    def tier(self) -> int:
        return 4

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

            analysis = _analyze_pawn_structure(board)
            parts = []

            for color_name in ("white", "black"):
                info = analysis[color_name]
                if info["doubled"]:
                    files = ", ".join(info["doubled"])
                    parts.append(f"{color_name.capitalize()} has doubled pawns on file(s) {files}.")
                if info["isolated"]:
                    sqs = ", ".join(info["isolated"])
                    parts.append(f"{color_name.capitalize()} has isolated pawn(s) on {sqs}.")
                if info["passed"]:
                    sqs = ", ".join(info["passed"])
                    parts.append(f"{color_name.capitalize()} has passed pawn(s) on {sqs}.")

            if not parts:
                answer = "No notable pawn structure features."
            else:
                answer = " ".join(parts)

            raw = {"fen": fen, "is_chess960": is_960}
            tpl = select_template(self.task_id(), self.rng)
            user_text = tpl.format(**raw)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


def _analyze_pawn_structure(board: chess.Board) -> dict:
    """Analyze pawn structure for both sides."""
    result = {
        "white": {"doubled": [], "isolated": [], "passed": []},
        "black": {"doubled": [], "isolated": [], "passed": []},
    }

    for color in (chess.WHITE, chess.BLACK):
        color_name = "white" if color == chess.WHITE else "black"
        opp_color = not color

        # Collect pawn squares by file
        pawns_by_file: dict[int, list[int]] = {}
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN and p.color == color:
                f = chess.square_file(sq)
                pawns_by_file.setdefault(f, []).append(sq)

        file_letters = "abcdefgh"

        # Doubled pawns: files with more than one pawn
        for f, sqs in pawns_by_file.items():
            if len(sqs) > 1:
                result[color_name]["doubled"].append(file_letters[f])

        # Isolated pawns: no friendly pawns on adjacent files
        for f, sqs in pawns_by_file.items():
            adj_files = [af for af in (f - 1, f + 1) if 0 <= af <= 7]
            has_neighbor = any(af in pawns_by_file for af in adj_files)
            if not has_neighbor:
                for sq in sqs:
                    result[color_name]["isolated"].append(chess.square_name(sq))

        # Passed pawns: no opponent pawns on same or adjacent files ahead
        for f, sqs in pawns_by_file.items():
            adj_files = [af for af in (f - 1, f, f + 1) if 0 <= af <= 7]
            for sq in sqs:
                rank = chess.square_rank(sq)
                is_passed = True
                if color == chess.WHITE:
                    ahead_ranks = range(rank + 1, 8)
                else:
                    ahead_ranks = range(0, rank)

                for af in adj_files:
                    for ar in ahead_ranks:
                        check_sq = chess.square(af, ar)
                        p = board.piece_at(check_sq)
                        if p and p.piece_type == chess.PAWN and p.color == opp_color:
                            is_passed = False
                            break
                    if not is_passed:
                        break

                if is_passed:
                    result[color_name]["passed"].append(chess.square_name(sq))

    return result
