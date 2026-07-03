"""Tier 7 verifier-first tasks."""

from __future__ import annotations

from typing import Generator, Iterator

import chess

from chess_llm.core.legality import format_legal_filter_trace_answer
from chess_llm.core.rays import SLIDER_RAY_DIRECTIONS, format_ray_walk_answer
from chess_llm.evals.benchmark import format_candidate_ratings_answer
from chess_llm.sft.context import board_from_raw
from chess_llm.sft.generators.base import TaskGenerator
from chess_llm.sft.generators.tier1_perception import _format_material_balance_trace_answer
from chess_llm.sft.generators.tier7_planning import _candidate_ratings_from_row
from chess_llm.sft.step_verification import (
    StepVerificationLabel,
    corrupt_candidate_ratings_best_line,
    corrupt_candidate_ratings_bucket_line,
    corrupt_candidate_ratings_eval_sign,
    corrupt_candidate_ratings_malformed_line,
    corrupt_final_move_set_line,
    corrupt_material_balance_line,
    format_step_verification_answer,
    number_trace_lines,
    sound_step_verification_label,
)


class StepVerification(TaskGenerator):
    """Task 7.9: find the single broken step in a numbered trace."""

    def task_id(self) -> str:
        return "7.9_step_verification"

    def tier(self) -> int:
        return 7

    def generate(self) -> Iterator[dict]:
        target = self.target_volume()
        count = 0

        for ev in self.config.get("candidate_rating_evals", []):
            raw = self.source_row(ev)
            if self.is_blocked(raw):
                continue
            board = board_from_raw(raw)
            if board is None:
                continue
            ratings = _valid_candidate_ratings(raw, board)
            if not ratings:
                continue

            clean_trace = format_candidate_ratings_answer(ratings)
            cases = _candidate_rating_cases(clean_trace)
            count = yield from self._emit_cases(
                raw,
                clean_trace=clean_trace,
                source_task="7.8_candidate_ratings",
                cases=cases,
                count=count,
                target=target,
            )
            if count >= target:
                return

        for entry in self.config.get("fen_pool", []):
            raw = self.source_row(entry)
            if self.is_blocked(raw):
                continue
            board = board_from_raw(raw)
            if board is None:
                continue
            for source_task, clean_trace, cases in _deterministic_trace_cases(board):
                count = yield from self._emit_cases(
                    raw,
                    clean_trace=clean_trace,
                    source_task=source_task,
                    cases=cases,
                    count=count,
                    target=target,
                )
                if count >= target:
                    return

    def _emit_cases(
        self,
        raw: dict,
        *,
        clean_trace: str,
        source_task: str,
        cases: list[tuple[str, StepVerificationLabel, str, str]],
        count: int,
        target: int,
    ) -> Generator[dict, None, int]:
        for trace, label, corruption_kind, difficulty in cases:
            if count >= target:
                return count
            case_raw = dict(raw)
            case_raw["verification_trace"] = number_trace_lines(trace)
            metadata = dict(case_raw.get("metadata", {}))
            metadata.update(
                {
                    "source": "step_verification",
                    "source_task": source_task,
                    "source_trace": clean_trace,
                    "displayed_trace": trace,
                    "corruption_kind": corruption_kind,
                    "difficulty": difficulty,
                    "verification_verdict": label.verdict,
                    "faulty_line": (
                        "none" if label.faulty_line is None else label.faulty_line
                    ),
                    "error_type": label.error_type,
                    "correction": label.correction,
                    "expected_answer": format_step_verification_answer(label),
                }
            )
            case_raw["metadata"] = metadata
            user_text = self.render_template(case_raw)
            yield self.format_example(
                case_raw,
                template_text=user_text,
                assistant_content=format_step_verification_answer(label),
            )
            count += 1
        return count


def _valid_candidate_ratings(raw: dict, board: chess.Board) -> list[dict[str, object]]:
    ratings = _candidate_ratings_from_row(raw)
    if len(ratings) < 5:
        return []
    ratings = ratings[:5]
    legal_moves = {move.uci() for move in board.legal_moves}
    if any(str(rating.get("uci", "")) not in legal_moves for rating in ratings):
        return []
    if any(
        rating.get("cp") is None and rating.get("mate") is None
        for rating in ratings
    ):
        return []
    return ratings


def _candidate_rating_cases(
    clean_trace: str,
) -> list[tuple[str, StepVerificationLabel, str, str]]:
    cases: list[tuple[str, StepVerificationLabel, str, str]] = []
    for corruptor, kind, difficulty in (
        (corrupt_candidate_ratings_best_line, "candidate_rating_wrong_best", "hard"),
        (corrupt_candidate_ratings_bucket_line, "candidate_rating_wrong_bucket", "hard"),
        (corrupt_candidate_ratings_eval_sign, "candidate_rating_wrong_eval", "hard"),
        (
            corrupt_candidate_ratings_malformed_line,
            "candidate_rating_malformed_line",
            "easy",
        ),
    ):
        corrupted = corruptor(clean_trace)
        if corrupted is None:
            continue
        trace, label = corrupted
        cases.append((trace, label, kind, difficulty))
    cases.append(
        (
            clean_trace,
            sound_step_verification_label(),
            "candidate_rating_sound",
            "sound",
        )
    )
    return cases


def _deterministic_trace_cases(
    board: chess.Board,
) -> list[tuple[str, str, list[tuple[str, StepVerificationLabel, str, str]]]]:
    traces: list[tuple[str, str, list[tuple[str, StepVerificationLabel, str, str]]]] = []

    material_trace = _format_material_balance_trace_answer(board)
    material_corrupted = corrupt_material_balance_line(material_trace)
    if material_corrupted is not None:
        traces.append(
            (
                "1.18_material_balance_trace",
                material_trace,
                [
                    (
                        material_corrupted[0],
                        material_corrupted[1],
                        "material_balance_wrong_eval",
                        "easy",
                    ),
                    (
                        material_trace,
                        sound_step_verification_label(),
                        "material_balance_sound",
                        "sound",
                    ),
                ],
            )
        )

    ray_square = _first_slider_square(board)
    if ray_square is not None:
        ray_trace = format_ray_walk_answer(board, ray_square)
        ray_corrupted = corrupt_final_move_set_line(
            ray_trace,
            prefix="Moves from rays:",
        )
        if ray_corrupted is not None:
            traces.append(
                (
                    "2.10_ray_walk",
                    ray_trace,
                    [
                        (
                            ray_corrupted[0],
                            ray_corrupted[1],
                            "ray_walk_wrong_move_set",
                            "medium",
                        ),
                        (
                            ray_trace,
                            sound_step_verification_label(),
                            "ray_walk_sound",
                            "sound",
                        ),
                    ],
                )
            )

    legal_filter_trace = format_legal_filter_trace_answer(board)
    if legal_filter_trace is not None:
        legal_filter_corrupted = corrupt_final_move_set_line(
            legal_filter_trace,
            prefix="All legal moves:",
        )
        if legal_filter_corrupted is not None:
            traces.append(
                (
                    "2.11_legal_filter_trace",
                    legal_filter_trace,
                    [
                        (
                            legal_filter_corrupted[0],
                            legal_filter_corrupted[1],
                            "legal_filter_wrong_move_set",
                            "medium",
                        ),
                        (
                            legal_filter_trace,
                            sound_step_verification_label(),
                            "legal_filter_sound",
                            "sound",
                        ),
                    ],
                )
            )
    return traces


def _first_slider_square(board: chess.Board) -> int | None:
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if (
            piece is not None
            and piece.color == board.turn
            and piece.piece_type in SLIDER_RAY_DIRECTIONS
        ):
            return square
    return None


__all__ = ["StepVerification"]
