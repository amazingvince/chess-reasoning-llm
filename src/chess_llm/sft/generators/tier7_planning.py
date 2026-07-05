"""Tier 7: Planning tasks (~170K examples).

7.1 Best move selection (<think>/<move> format, depth >= 30)
7.2 Puzzle solving (Lichess puzzles rated 1000-2500)
7.3 Move consequence (PV line analysis)
7.8 Candidate ratings (fixed grammar, true MultiPV only)
7.10 Best-line trace (fixed grammar, true MultiPV PV only)
7.11 History-conditioned best move (PGN prefix + FEN -> next move)
"""

from __future__ import annotations

from typing import Iterator

import chess

from chess_llm.sft.context import board_from_raw, raw_is_chess960
from chess_llm.evals.benchmark import format_candidate_ratings_answer
from chess_llm.sft.best_line_trace import best_line_trace_payload
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


class HistoryBestMoveSelection(TaskGenerator):
    """Task 7.11: next-move imitation from a game prefix and current FEN."""

    def task_id(self) -> str:
        return "7.11_history_best_move"

    def tier(self) -> int:
        return 7

    def generate(self) -> Iterator[dict]:
        positions = self.config.get("game_positions", [])
        target = self.target_volume()
        count = 0

        for position in positions:
            if count >= target:
                return
            raw = self.source_row(position)
            if self.is_blocked(raw):
                continue

            move_history = str(position.get("move_history", "") or "").strip()
            target_move = str(position.get("move_played_uci", "") or "").strip().lower()
            if not move_history or not target_move:
                continue

            board = board_from_raw(raw)
            if board is None:
                continue
            try:
                move = chess.Move.from_uci(target_move)
            except ValueError:
                continue
            if move not in board.legal_moves:
                continue

            metadata = dict(raw.get("metadata", {}))
            metadata.update(
                {
                    "source": "lichess_game_history",
                    "target_move": target_move,
                    "move_history": move_history,
                    "game_id": position.get("game_id", ""),
                    "ply": position.get("ply"),
                    "game_phase": position.get("game_phase", ""),
                }
            )
            raw.update(
                {
                    "move_history": move_history,
                    "target_move": target_move,
                    "metadata": metadata,
                }
            )
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(
                raw,
                template_text=user_text,
                assistant_content=f"<move>{target_move}</move>",
            )
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


class CandidateRatings(TaskGenerator):
    """Task 7.8: rate exactly five legal MultiPV candidate moves."""

    def task_id(self) -> str:
        return "7.8_candidate_ratings"

    def tier(self) -> int:
        return 7

    def generate(self) -> Iterator[dict]:
        evals = self.config.get("candidate_rating_evals", [])
        target = self.target_volume()
        count = 0

        for ev in evals:
            if count >= target:
                return
            raw = self.source_row(ev)
            if self.is_blocked(raw):
                continue

            board = board_from_raw(raw)
            if board is None:
                continue
            ratings = _candidate_ratings_from_row(raw)
            if len(ratings) < 5:
                continue
            ratings = ratings[:5]
            legal_moves = {move.uci() for move in board.legal_moves}
            if any(str(rating["uci"]) not in legal_moves for rating in ratings):
                continue
            if any(
                rating.get("cp") is None and rating.get("mate") is None
                for rating in ratings
            ):
                continue

            candidate_moves = " ".join(str(rating["uci"]) for rating in ratings)
            raw["candidate_moves"] = candidate_moves
            metadata = dict(raw.get("metadata", {}))
            metadata.update({
                "source": "stockfish_multipv",
                "target_move": ratings[0]["uci"],
                "candidate_moves": candidate_moves,
                "candidate_ratings": ratings,
                "multipv_depth": raw.get("multipv_depth", raw.get("depth", 0)),
            })
            raw["metadata"] = metadata
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(
                raw,
                template_text=user_text,
                assistant_content=format_candidate_ratings_answer(ratings),
            )
            count += 1


class BestLineTrace(TaskGenerator):
    """Task 7.10: seeded fixed-grammar trace for the engine best PV."""

    def task_id(self) -> str:
        return "7.10_best_line_trace"

    def tier(self) -> int:
        return 7

    def generate(self) -> Iterator[dict]:
        evals = self.config.get("candidate_rating_evals", [])
        target = self.target_volume()
        count = 0

        for ev in evals:
            if count >= target:
                return
            raw = self.source_row(ev)
            if self.is_blocked(raw):
                continue

            board = board_from_raw(raw)
            if board is None:
                continue
            ratings = _candidate_ratings_from_row(raw)
            if not ratings:
                continue
            payload = best_line_trace_payload(
                str(raw.get("fen", "")),
                ratings[0],
                chess960=raw_is_chess960(raw),
            )
            if payload is None:
                continue

            best_move = str(payload["best_move"])
            raw["move"] = best_move
            raw["best_move"] = best_move
            raw["pv_line"] = str(payload["pv_line"])
            metadata = dict(raw.get("metadata", {}))
            metadata.update({
                "source": "stockfish_multipv",
                "source_task": "7.8_candidate_ratings",
                "target_move": best_move,
                "best_move": best_move,
                "pv": payload["pv"],
                "pv_line": payload["pv_line"],
                "cp": payload["cp"],
                "mate": payload["mate"],
                "bucket": payload["bucket"],
                "expected_answer": payload["expected_answer"],
                "multipv_depth": raw.get("multipv_depth", raw.get("depth", 0)),
                "multipv_k": raw.get("multipv_k"),
                "pv_len": raw.get("pv_len"),
            })
            raw["metadata"] = metadata
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(
                raw,
                template_text=user_text,
                assistant_content=str(payload["expected_answer"]),
            )
            count += 1


def _candidate_ratings_from_row(raw: dict) -> list[dict[str, object]]:
    source = raw.get("candidate_ratings")
    if not isinstance(source, list):
        source = raw.get("move_evaluations")
    if not isinstance(source, list):
        return []
    ratings: list[dict[str, object]] = []
    for item in source:
        if not isinstance(item, dict):
            continue
        uci = item.get("uci") or item.get("move") or item.get("best_move")
        if not isinstance(uci, str) or not uci:
            continue
        cp = item.get("cp", item.get("centipawn"))
        mate = item.get("mate", item.get("mate_in"))
        rating: dict[str, object] = {"uci": uci.lower()}
        if cp is not None:
            try:
                rating["cp"] = int(cp)
            except (TypeError, ValueError):
                pass
        if mate is not None:
            try:
                rating["mate"] = int(mate)
            except (TypeError, ValueError):
                pass
        pv = item.get("pv") or item.get("pv_line") or item.get("principal_variation")
        if pv:
            rating["pv_line"] = " ".join(pv) if isinstance(pv, list) else str(pv)
        if item.get("rank") is not None:
            rating["rank"] = item.get("rank")
        if item.get("expectation") is not None:
            rating["expectation"] = item.get("expectation")
        ratings.append(rating)
    return ratings
