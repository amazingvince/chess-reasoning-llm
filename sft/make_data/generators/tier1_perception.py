"""Tier 1: Perception tasks (~440K examples).

1.1 FEN -> ASCII board diagram
1.2 ASCII board -> FEN
1.3 Piece identification (what's on a square / where are the pieces)
1.4 Piece counting (material count, balance)
1.5 State tracking (apply moves, report resulting position)
"""

from __future__ import annotations

from random import Random
from typing import Iterator

import chess

from config.templates import TEMPLATES, select_template
from generators.base import TaskGenerator


def _board_to_ascii(board: chess.Board) -> str:
    """Render a board as an ASCII diagram with rank/file labels."""
    lines = []
    for rank in range(7, -1, -1):
        row = []
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            row.append(piece.symbol() if piece else ".")
        lines.append(f"{rank + 1} {' '.join(row)}")
    lines.append("  a b c d e f g h")
    return "\n".join(lines)


_PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}
_PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


class FENToBoard(TaskGenerator):
    """Task 1.1: FEN -> ASCII board diagram."""

    def task_id(self) -> str:
        return "1.1_fen_to_board"

    def tier(self) -> int:
        return 1

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
            ascii_board = _board_to_ascii(board)

            raw = {"fen": fen, "is_chess960": is_960, "board": ascii_board}
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
            ascii_board = _board_to_ascii(board)

            raw = {"fen": fen, "board": ascii_board, "is_chess960": is_960}
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=fen)
            count += 1


class PieceIdentification(TaskGenerator):
    """Task 1.3: Identify pieces on squares or locate specific pieces."""

    def task_id(self) -> str:
        return "1.3_piece_identification"

    def tier(self) -> int:
        return 1

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

            # Randomly choose subtask: what's on square, or where are pieces
            if self.rng.random() < 0.5:
                # What is on a specific square?
                sq = self.rng.choice(chess.SQUARES)
                square_name = chess.square_name(sq)
                piece = board.piece_at(sq)
                if piece:
                    answer = f"{('white' if piece.color == chess.WHITE else 'black')} {_PIECE_NAMES[piece.piece_type]}"
                else:
                    answer = "empty"

                raw = {"fen": fen, "square": square_name, "is_chess960": is_960,
                       "color": "", "piece": ""}
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

                raw = {"fen": fen, "color": color_name, "piece": piece_name,
                       "is_chess960": is_960, "square": ""}
                templates_for_locate = [
                    t for t in TEMPLATES[self.task_id()]
                    if "{color}" in t and "{piece}" in t and "{square}" not in t
                ]
                user_text = self.render_template(raw, self.rng.choice(templates_for_locate))
                yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class PieceCounting(TaskGenerator):
    """Task 1.4: Count pieces, material balance."""

    def task_id(self) -> str:
        return "1.4_piece_counting"

    def tier(self) -> int:
        return 1

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

            color = self.rng.choice(["white", "black"])
            piece_type_name = self.rng.choice(
                ["pawn", "knight", "bishop", "rook", "queen", "piece"]
            )
            raw = {"fen": fen, "color": color, "piece": piece_type_name,
                   "is_chess960": is_960}
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)

            # Branch the answer based on the selected template
            tpl_lower = tpl.lower()
            if "how many pieces does {color}" in tpl_lower:
                # Answer only for the requested color
                if color == "white":
                    answer = f"White has {white_total} piece(s): {', '.join(w_parts)}."
                else:
                    answer = f"Black has {black_total} piece(s): {', '.join(b_parts)}."
            elif "how many {piece}s" in tpl_lower:
                # Count only the requested piece type
                _name_to_type = {
                    "pawn": chess.PAWN, "knight": chess.KNIGHT,
                    "bishop": chess.BISHOP, "rook": chess.ROOK,
                    "queen": chess.QUEEN, "piece": None,
                }
                pt = _name_to_type.get(piece_type_name)
                if pt is not None:
                    wc = white_counts.get(piece_type_name, 0)
                    bc = black_counts.get(piece_type_name, 0)
                    answer = (
                        f"White has {wc} {piece_type_name}(s), "
                        f"black has {bc} {piece_type_name}(s). "
                        f"Total: {wc + bc}."
                    )
                else:
                    answer = (
                        f"White ({white_total} pieces): {', '.join(w_parts)}. "
                        f"Black ({black_total} pieces): {', '.join(b_parts)}. "
                        f"{balance_str}"
                    )
            elif "minor pieces" in tpl_lower:
                # Count only knights and bishops
                w_minor = white_counts.get("knight", 0) + white_counts.get("bishop", 0)
                b_minor = black_counts.get("knight", 0) + black_counts.get("bishop", 0)
                if color == "white":
                    answer = f"White has {w_minor} minor piece(s) ({white_counts.get('knight', 0)} knight(s), {white_counts.get('bishop', 0)} bishop(s))."
                else:
                    answer = f"Black has {b_minor} minor piece(s) ({black_counts.get('knight', 0)} knight(s), {black_counts.get('bishop', 0)} bishop(s))."
            else:
                # Full material count for both sides (generic templates)
                answer = (
                    f"White ({white_total} pieces): {', '.join(w_parts)}. "
                    f"Black ({black_total} pieces): {', '.join(b_parts)}. "
                    f"{balance_str}"
                )

            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class StateTracking(TaskGenerator):
    """Task 1.5: Apply 1-8 moves, track resulting position."""

    def task_id(self) -> str:
        return "1.5_state_tracking"

    def tier(self) -> int:
        return 1

    def generate(self) -> Iterator[dict]:
        game_positions = self.config.get("game_positions", [])
        target = self.target_volume()
        count = 0

        for pos in game_positions:
            if count >= target:
                return
            fen = pos.get("fen", "")
            if self.is_blocked(fen):
                continue

            is_960 = pos.get("is_chess960", False)
            board = chess.Board(fen, chess960=is_960)
            n_moves = self.rng.randint(1, 8)

            # Play random legal moves
            moves_played = []
            for _ in range(n_moves):
                legal = list(board.legal_moves)
                if not legal:
                    break
                move = self.rng.choice(legal)
                moves_played.append(move.uci())
                board.push(move)

            if not moves_played:
                continue

            result_fen = board.fen()
            moves_str = " ".join(moves_played)

            raw = {
                "fen": fen,
                "moves": moves_str,
                "n_moves": str(len(moves_played)),
                "is_chess960": is_960,
                "metadata": {"result_fen": result_fen, "moves": moves_str},
            }
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(
                raw, template_text=user_text, assistant_content=result_fen
            )
            count += 1
