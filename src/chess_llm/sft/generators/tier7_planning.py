"""Tier 7: Planning tasks (~170K examples).

7.1 Best move selection (<think>/<move> format, depth >= 30)
7.2 Puzzle solving (Lichess puzzles rated 1000-2500)
7.3 Move consequence (PV line analysis)
"""

from __future__ import annotations

from typing import Iterator

import chess

from chess_llm.sft.context import board_from_raw
from chess_llm.sft.generators.base import TaskGenerator
from chess_llm.sft.generators.reasoning_traces import (
    generate_endgame_trace,
    generate_positional_trace,
    generate_tactical_trace,
)
from chess_llm.sft.templates import select_template


class BestMoveSelection(TaskGenerator):
    """Task 7.1: Best move with <think>/<move> format.

    Uses Position Evals depth >= 30 as the source.  MATE dataset rows
    (better of exactly two candidate moves) are intentionally excluded:
    a pairwise comparison is not an answer to "What is the best move?",
    so they need a dedicated A-vs-B comparison task instead.
    """

    def task_id(self) -> str:
        return "7.1_best_move_selection"

    def tier(self) -> int:
        return 7

    def generate(self) -> Iterator[dict]:
        evals = self.config.get("best_move_evals", [])
        target = self.target_volume()
        count = 0

        for ev in evals:
            if count >= target:
                return
            raw = self.source_row(ev)
            fen = raw["fen"]
            if self.is_blocked(raw):
                continue

            best_move = ev.get("best_move", "")
            if not best_move:
                continue

            board = board_from_raw(raw)
            if board is None:
                continue
            try:
                m = chess.Move.from_uci(best_move)
                if m not in board.legal_moves:
                    continue
            except ValueError:
                continue

            cp = ev.get("cp")
            mate = ev.get("mate")
            pv_line = ev.get("pv_line", "")

            side_to_move_is_mating = (
                mate is not None and (mate > 0) == (board.turn == chess.WHITE)
            )
            if side_to_move_is_mating:
                trace = generate_tactical_trace(
                    board, best_move, ["mate"], pv_line, self.rng
                )
            elif mate is not None:
                # The side to move is defending against mate; do not
                # narrate its move as a mating pattern.
                trace = generate_positional_trace(board, best_move, cp, self.rng)
            elif cp is not None and abs(cp) > 300:
                trace = generate_tactical_trace(
                    board, best_move, [], pv_line, self.rng
                )
            else:
                trace = generate_positional_trace(board, best_move, cp, self.rng)

            metadata = dict(raw.get("metadata", {}))
            metadata.update({
                "source": "lichess_evals",
                "target_move": best_move,
                "stockfish_eval_cp": cp,
                "stockfish_eval_mate": mate,
                "depth": ev.get("depth", 0),
            })
            raw["metadata"] = metadata
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=trace)
            count += 1


class PuzzleSolving(TaskGenerator):
    """Task 7.2: Lichess puzzles rated 1000-2500."""

    def task_id(self) -> str:
        return "7.2_puzzle_solving"

    def tier(self) -> int:
        return 7

    def generate(self) -> Iterator[dict]:
        puzzles = self.config.get("puzzles", [])
        target = self.target_volume()
        count = 0

        for puzzle in puzzles:
            if count >= target:
                return

            raw = self.source_row(puzzle)
            fen = raw["fen"]
            if self.is_blocked(raw):
                continue

            rating = puzzle.get("rating", 0)
            if rating < 1000 or rating > 2500:
                continue

            solution = puzzle.get("solution_first_move", "")
            if not solution:
                continue

            themes = puzzle.get("themes", [])
            solution_full = puzzle.get("solution_full", [])
            pv_line = " ".join(solution_full) if solution_full else solution

            board = board_from_raw(raw)
            if board is None:
                continue
            trace = generate_tactical_trace(
                board, solution, themes, pv_line, self.rng
            )

            side = puzzle.get("side", "white" if board.turn == chess.WHITE else "black")
            raw["side"] = side
            metadata = dict(raw.get("metadata", {}))
            metadata.update({
                "source": "lichess_puzzles",
                "target_move": solution,
                "rating": rating,
                "themes": themes,
                "puzzle_id": puzzle.get("puzzle_id", ""),
            })
            raw["metadata"] = metadata
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=trace)
            count += 1


class MoveConsequence(TaskGenerator):
    """Task 7.3: PV line analysis from Position Evals + Lichess games."""

    def task_id(self) -> str:
        return "7.3_move_consequence"

    def tier(self) -> int:
        return 7

    def generate(self) -> Iterator[dict]:
        evals = self.config.get("consequence_evals", [])
        target = self.target_volume()
        count = 0

        for ev in evals:
            if count >= target:
                return
            raw = self.source_row(ev)
            fen = raw["fen"]
            if self.is_blocked(raw):
                continue

            best_move = ev.get("best_move", "")
            pv_line = ev.get("pv_line", "")
            if not best_move or not pv_line:
                continue

            board = board_from_raw(raw)
            if board is None:
                continue
            try:
                m = chess.Move.from_uci(best_move)
                if m not in board.legal_moves:
                    continue
            except ValueError:
                continue

            pv_moves = pv_line.split()
            cp = ev.get("cp")

            # Build consequence analysis
            thoughts = []
            thoughts.append(f"After {best_move}:")

            # Replay PV and describe key moments
            analysis_board = board.copy()
            for i, uci in enumerate(pv_moves[:6]):
                try:
                    mv = chess.Move.from_uci(uci)
                    if mv not in analysis_board.legal_moves:
                        break
                    if analysis_board.is_capture(mv):
                        captured = analysis_board.piece_at(mv.to_square)
                        if captured:
                            piece_names = {
                                chess.PAWN: "pawn", chess.KNIGHT: "knight",
                                chess.BISHOP: "bishop", chess.ROOK: "rook",
                                chess.QUEEN: "queen", chess.KING: "king",
                            }
                            cap_name = piece_names.get(captured.piece_type, "piece")
                            thoughts.append(f"{uci} captures the {cap_name}.")
                    analysis_board.push(mv)
                    if analysis_board.is_check():
                        thoughts.append(f"{uci} gives check.")
                except (ValueError, chess.InvalidMoveError):
                    break

            if cp is not None:
                if abs(cp) < 50:
                    thoughts.append("The position remains roughly equal.")
                elif cp > 0:
                    thoughts.append(f"White maintains an advantage of {cp/100:.1f} pawns.")
                else:
                    thoughts.append(f"Black maintains an advantage of {abs(cp)/100:.1f} pawns.")

            pv_display = " ".join(pv_moves[:5])
            thoughts.append(f"Expected line: {pv_display}")
            answer = (
                f"<think>{' '.join(thoughts)}</think>\n"
                f"<move>{best_move}</move>"
            )

            raw["move"] = best_move
            metadata = dict(raw.get("metadata", {}))
            metadata.update({
                "source": "lichess_evals",
                "target_move": best_move,
                "pv_line": pv_line,
                "stockfish_eval_cp": cp,
            })
            raw["metadata"] = metadata
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1
