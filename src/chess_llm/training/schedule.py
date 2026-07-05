"""Time-varying tier-mix schedule for a single long SFT run.

This module is intentionally stdlib-only: the training entrypoints import it
(lazily) and must keep importing without torch/transformers/trl/datasets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

MIN_TIER = 1
MAX_TIER = 7
DEFAULT_EFFECTIVE_BATCH = 32


@dataclass(frozen=True)
class ScheduleSegment:
    """One contiguous slice of the run with a fixed tier mix."""

    name: str
    fraction_of_run: float
    tier_weights: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class ScheduleConfig:
    """A full-run schedule; duck-types PhaseConfig for the training harness.

    ``build_sft_config``/``estimate_training_steps``/``resolve_checkpoint``
    only touch ``name``/``display_name``/``epochs``/``learning_rate``/
    ``warmup_ratio``/``weight_decay``/``resume_from``, all present here.
    """

    name: str
    display_name: str
    segments: tuple[ScheduleSegment, ...]
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    epochs: int = 1
    resume_from: str | None = None


@dataclass(frozen=True)
class SegmentPlan:
    """Concrete row allocation for one segment of the run."""

    index: int
    name: str
    start_row: int
    end_row: int
    tier_weights: tuple[tuple[int, float], ...]
    tier_rows: tuple[tuple[int, int], ...]


def validate_schedule(schedule: ScheduleConfig) -> None:
    """Raise ValueError with a precise message for any malformed schedule."""
    if not schedule.segments:
        raise ValueError(f"Schedule {schedule.name!r} has no segments")

    total_fraction = 0.0
    for segment in schedule.segments:
        if segment.fraction_of_run <= 0:
            raise ValueError(
                f"Schedule {schedule.name!r} segment {segment.name!r}: "
                f"fraction_of_run must be > 0, got {segment.fraction_of_run}"
            )
        total_fraction += segment.fraction_of_run

        if not segment.tier_weights:
            raise ValueError(
                f"Schedule {schedule.name!r} segment {segment.name!r} has no tier weights"
            )
        seen_tiers: set[int] = set()
        n_positive = 0
        for tier, weight in segment.tier_weights:
            if not (MIN_TIER <= tier <= MAX_TIER):
                raise ValueError(
                    f"Schedule {schedule.name!r} segment {segment.name!r}: "
                    f"tier {tier} is outside {MIN_TIER}..{MAX_TIER}"
                )
            if tier in seen_tiers:
                raise ValueError(
                    f"Schedule {schedule.name!r} segment {segment.name!r}: "
                    f"tier {tier} is listed more than once"
                )
            seen_tiers.add(tier)
            if weight < 0:
                raise ValueError(
                    f"Schedule {schedule.name!r} segment {segment.name!r}: "
                    f"tier {tier} weight must be >= 0, got {weight}"
                )
            if weight > 0:
                n_positive += 1
        if n_positive == 0:
            raise ValueError(
                f"Schedule {schedule.name!r} segment {segment.name!r} "
                "needs at least one positive tier weight"
            )

    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError(
            f"Schedule {schedule.name!r} segment fractions must sum to 1.0, "
            f"got {total_fraction!r}"
        )


def schedule_tiers(segments: Sequence[ScheduleSegment]) -> list[int]:
    """Sorted union of tiers with a positive weight in any segment."""
    return sorted({
        tier
        for segment in segments
        for tier, weight in segment.tier_weights
        if weight > 0
    })


def _largest_remainder(total: int, weights: Sequence[float]) -> list[int]:
    """Split ``total`` into integer counts proportional to ``weights`` exactly."""
    weight_sum = sum(weights)
    if total < 0 or weight_sum <= 0:
        raise ValueError(
            f"Cannot allocate {total} rows across weights summing to {weight_sum}"
        )
    quotas = [total * weight / weight_sum for weight in weights]
    counts = [math.floor(quota) for quota in quotas]
    remainder = total - sum(counts)
    by_fraction = sorted(
        range(len(weights)),
        key=lambda i: (-(quotas[i] - counts[i]), i),
    )
    while remainder > 0:
        for i in by_fraction:
            if remainder == 0:
                break
            counts[i] += 1
            remainder -= 1
    return counts


def plan_segments(
    segments: Sequence[ScheduleSegment],
    total_examples: int,
) -> list[SegmentPlan]:
    """Allocate ``total_examples`` rows across segments and tiers exactly.

    Uses largest-remainder apportionment twice — once for segment sizes,
    once for per-tier rows within each segment — so both allocations sum
    exactly to their targets.
    """
    if total_examples <= 0:
        raise ValueError(f"total_examples must be > 0, got {total_examples}")
    if not segments:
        raise ValueError("plan_segments requires at least one segment")

    segment_rows = _largest_remainder(
        total_examples,
        [segment.fraction_of_run for segment in segments],
    )

    plans: list[SegmentPlan] = []
    start_row = 0
    for index, (segment, n_rows) in enumerate(zip(segments, segment_rows)):
        positive = [(tier, weight) for tier, weight in segment.tier_weights if weight > 0]
        tier_counts = _largest_remainder(n_rows, [weight for _, weight in positive])
        end_row = start_row + n_rows
        plans.append(
            SegmentPlan(
                index=index,
                name=segment.name,
                start_row=start_row,
                end_row=end_row,
                tier_weights=segment.tier_weights,
                tier_rows=tuple(
                    (tier, count) for (tier, _), count in zip(positive, tier_counts)
                ),
            )
        )
        start_row = end_row
    return plans


def boundary_steps(
    plans: Sequence[SegmentPlan],
    effective_batch: int = DEFAULT_EFFECTIVE_BATCH,
) -> list[int]:
    """Optimizer step at which each segment ends (remainder rows round up)."""
    if effective_batch <= 0:
        raise ValueError(f"effective_batch must be > 0, got {effective_batch}")
    return [math.ceil(plan.end_row / effective_batch) for plan in plans]


def segment_index_at_step(step: int, boundaries: Sequence[int]) -> int:
    """Index of the segment covering optimizer ``step`` (clamped to the last)."""
    if not boundaries:
        raise ValueError("segment_index_at_step requires at least one boundary")
    for index, boundary in enumerate(boundaries):
        if step <= boundary:
            return index
    return len(boundaries) - 1


def replay_fraction_t12(weights: Sequence[tuple[int, float]]) -> float:
    """Fraction of a segment's mix devoted to Tier 1/2 replay."""
    total = sum(weight for _, weight in weights)
    if total <= 0:
        return 0.0
    return sum(weight for tier, weight in weights if tier in (1, 2)) / total


SCHEDULE_V1 = ScheduleConfig(
    name="schedule",
    display_name="Schedule v1: single-run time-varying tier mix",
    segments=(
        ScheduleSegment(
            name="mechanics",
            fraction_of_run=0.25,
            tier_weights=((1, 0.50), (2, 0.50)),
        ),
        ScheduleSegment(
            name="broaden",
            fraction_of_run=0.35,
            tier_weights=(
                (1, 0.125),
                (2, 0.125),
                (3, 0.20),
                (4, 0.20),
                (5, 0.15),
                (6, 0.20),
            ),
        ),
        ScheduleSegment(
            name="consolidate",
            fraction_of_run=0.25,
            tier_weights=(
                (1, 0.10),
                (2, 0.10),
                (3, 0.15),
                (4, 0.15),
                (5, 0.15),
                (6, 0.15),
                (7, 0.20),
            ),
        ),
        ScheduleSegment(
            name="planning",
            fraction_of_run=0.15,
            tier_weights=(
                (1, 0.10),
                (2, 0.10),
                (3, 0.10),
                (4, 0.10),
                (5, 0.10),
                (6, 0.10),
                (7, 0.40),
            ),
        ),
    ),
    learning_rate=2e-5,
    warmup_ratio=0.03,
    weight_decay=0.01,
)


def _bc_move_choice_probe_schedule(
    *,
    name: str,
    replay_fraction: float,
) -> ScheduleConfig:
    """Build a one-pass Tier 7 move-choice probe with Phase A/B replay."""
    if not (0.0 <= replay_fraction < 1.0):
        raise ValueError(f"replay_fraction must be in [0, 1), got {replay_fraction}")
    tier7_fraction = 1.0 - replay_fraction
    replay_tier_fraction = replay_fraction / 6.0
    replay_weights = (
        tuple((tier, replay_tier_fraction) for tier in range(1, 7))
        if replay_fraction > 0.0
        else ()
    )
    percent = int(round(replay_fraction * 100))
    return ScheduleConfig(
        name=name,
        display_name=(
            "BC move-choice probe: "
            f"Tier 7 move-choice with {percent}% Phase A/B replay"
        ),
        segments=(
            ScheduleSegment(
                name="move_choice",
                fraction_of_run=1.0,
                tier_weights=(*replay_weights, (7, tier7_fraction)),
            ),
        ),
        learning_rate=2e-6,
        warmup_ratio=0.03,
        weight_decay=0.01,
    )


BC_PROBE_R0 = _bc_move_choice_probe_schedule(
    name="bc-probe-r0",
    replay_fraction=0.0,
)
BC_PROBE_R10 = _bc_move_choice_probe_schedule(
    name="bc-probe-r10",
    replay_fraction=0.10,
)
BC_PROBE_R25 = _bc_move_choice_probe_schedule(
    name="bc-probe-r25",
    replay_fraction=0.25,
)

SCHEDULES: dict[str, ScheduleConfig] = {
    "schedule": SCHEDULE_V1,
    "bc-probe-r0": BC_PROBE_R0,
    "bc-probe-r10": BC_PROBE_R10,
    "bc-probe-r25": BC_PROBE_R25,
}


__all__ = [
    "BC_PROBE_R0",
    "BC_PROBE_R10",
    "BC_PROBE_R25",
    "DEFAULT_EFFECTIVE_BATCH",
    "SCHEDULE_V1",
    "SCHEDULES",
    "ScheduleConfig",
    "ScheduleSegment",
    "SegmentPlan",
    "boundary_steps",
    "plan_segments",
    "replay_fraction_t12",
    "schedule_tiers",
    "segment_index_at_step",
    "validate_schedule",
]
