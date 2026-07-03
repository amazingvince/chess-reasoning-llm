"""Tier 7 verifier-first tasks."""

from __future__ import annotations

from typing import Iterator

import chess

from chess_llm.evals.benchmark import format_candidate_ratings_answer
from chess_llm.sft.context import board_from_raw
from chess_llm.sft.generators.base import TaskGenerator
from chess_llm.sft.generators.tier7_planning import _candidate_ratings_from_row
from chess_llm.sft.step_verification import (
    StepVerificationLabel,
    corrupt_candidate_ratings_best_line,
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
            if count >= target:
                return
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
            cases: list[tuple[str, StepVerificationLabel, str]] = []
            corrupted = corrupt_candidate_ratings_best_line(clean_trace)
            if corrupted is not None:
                bad_trace, bad_label = corrupted
                cases.append((bad_trace, bad_label, "candidate_rating_wrong_best"))
            cases.append(
                (clean_trace, sound_step_verification_label(), "candidate_rating_sound")
            )

            for trace, label, corruption_kind in cases:
                if count >= target:
                    return
                case_raw = dict(raw)
                case_raw["verification_trace"] = number_trace_lines(trace)
                metadata = dict(case_raw.get("metadata", {}))
                metadata.update(
                    {
                        "source": "step_verification",
                        "source_task": "7.8_candidate_ratings",
                        "source_trace": clean_trace,
                        "displayed_trace": trace,
                        "corruption_kind": corruption_kind,
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


__all__ = ["StepVerification"]
