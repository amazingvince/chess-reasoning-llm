"""Tests for the stdlib-only training schedule module."""

from __future__ import annotations

import pytest

from chess_llm.training.schedule import (
    SCHEDULE_V1,
    SCHEDULES,
    ScheduleConfig,
    ScheduleSegment,
    boundary_steps,
    plan_segments,
    replay_fraction_t12,
    schedule_tiers,
    segment_index_at_step,
    validate_schedule,
)


def _schedule(*segments: ScheduleSegment) -> ScheduleConfig:
    return ScheduleConfig(
        name="schedule-test",
        display_name="Test schedule",
        segments=tuple(segments),
        learning_rate=1e-4,
        warmup_ratio=0.0,
        weight_decay=0.0,
    )


def _segment(name: str, fraction: float, weights) -> ScheduleSegment:
    return ScheduleSegment(
        name=name,
        fraction_of_run=fraction,
        tier_weights=tuple(weights),
    )


def test_default_schedule_is_valid_and_registered():
    validate_schedule(SCHEDULE_V1)

    assert SCHEDULES["schedule"] is SCHEDULE_V1
    assert SCHEDULE_V1.name == "schedule"
    assert SCHEDULE_V1.epochs == 1
    assert SCHEDULE_V1.resume_from is None
    assert SCHEDULE_V1.learning_rate == 2e-5
    assert SCHEDULE_V1.warmup_ratio == 0.03
    assert SCHEDULE_V1.weight_decay == 0.01
    assert abs(sum(s.fraction_of_run for s in SCHEDULE_V1.segments) - 1.0) <= 1e-6
    assert schedule_tiers(SCHEDULE_V1.segments) == [1, 2, 3, 4, 5, 6, 7]


def test_default_schedule_keeps_t12_replay_floor_after_first_segment():
    first, *rest = SCHEDULE_V1.segments
    assert replay_fraction_t12(first.tier_weights) == 1.0
    for segment in rest:
        replay = replay_fraction_t12(segment.tier_weights)
        assert 0.20 <= replay <= 0.30, (segment.name, replay)


def test_validate_schedule_rejects_fractions_not_summing_to_one():
    schedule = _schedule(
        _segment("s0", 0.5, [(1, 1.0)]),
        _segment("s1", 0.4, [(2, 1.0)]),
    )
    with pytest.raises(ValueError, match="fractions must sum to 1.0"):
        validate_schedule(schedule)


def test_validate_schedule_rejects_nonpositive_fraction():
    schedule = _schedule(
        _segment("s0", 0.0, [(1, 1.0)]),
        _segment("s1", 1.0, [(2, 1.0)]),
    )
    with pytest.raises(ValueError, match="fraction_of_run must be > 0"):
        validate_schedule(schedule)


def test_validate_schedule_rejects_tier_out_of_range():
    for bad_tier in (0, 8):
        schedule = _schedule(_segment("s0", 1.0, [(bad_tier, 1.0)]))
        with pytest.raises(ValueError, match=f"tier {bad_tier} is outside 1..7"):
            validate_schedule(schedule)


def test_validate_schedule_rejects_negative_weight():
    schedule = _schedule(_segment("s0", 1.0, [(1, 1.0), (2, -0.1)]))
    with pytest.raises(ValueError, match="weight must be >= 0"):
        validate_schedule(schedule)


def test_validate_schedule_rejects_segment_with_no_positive_weight():
    schedule = _schedule(_segment("s0", 1.0, [(1, 0.0), (2, 0.0)]))
    with pytest.raises(ValueError, match="at least one positive tier weight"):
        validate_schedule(schedule)


def test_validate_schedule_rejects_duplicate_tier_in_segment():
    schedule = _schedule(_segment("s0", 1.0, [(1, 0.5), (1, 0.5)]))
    with pytest.raises(ValueError, match="listed more than once"):
        validate_schedule(schedule)


def test_validate_schedule_rejects_empty_schedule():
    with pytest.raises(ValueError, match="has no segments"):
        validate_schedule(_schedule())


def test_plan_segments_largest_remainder_gives_exact_segment_totals():
    segments = (
        _segment("s0", 1 / 3, [(1, 1.0)]),
        _segment("s1", 1 / 3, [(2, 1.0)]),
        _segment("s2", 1 / 3, [(3, 1.0)]),
    )

    plans = plan_segments(segments, 100)

    rows = [plan.end_row - plan.start_row for plan in plans]
    assert rows == [34, 33, 33]
    assert plans[0].start_row == 0
    assert all(
        plans[i].end_row == plans[i + 1].start_row for i in range(len(plans) - 1)
    )
    assert plans[-1].end_row == 100


def test_plan_segments_largest_remainder_gives_exact_tier_totals():
    segments = (
        _segment("s0", 1.0, [(1, 0.5), (2, 0.3), (3, 0.2)]),
    )

    plans = plan_segments(segments, 7)

    assert plans[0].tier_rows == ((1, 4), (2, 2), (3, 1))
    assert sum(rows for _, rows in plans[0].tier_rows) == 7


def test_plan_segments_default_schedule_exact_for_awkward_total():
    total = 999_983  # prime, so every fraction leaves a remainder
    plans = plan_segments(SCHEDULE_V1.segments, total)

    assert sum(plan.end_row - plan.start_row for plan in plans) == total
    assert plans[-1].end_row == total
    for plan in plans:
        assert sum(rows for _, rows in plan.tier_rows) == plan.end_row - plan.start_row
        assert all(rows >= 0 for _, rows in plan.tier_rows)


def test_plan_segments_omits_zero_weight_tiers_from_tier_rows():
    plans = plan_segments(
        (_segment("s0", 1.0, [(1, 1.0), (2, 0.0)]),),
        10,
    )

    assert plans[0].tier_rows == ((1, 10),)


def test_plan_segments_rejects_nonpositive_total():
    with pytest.raises(ValueError, match="total_examples must be > 0"):
        plan_segments((_segment("s0", 1.0, [(1, 1.0)]),), 0)


def test_boundary_steps_round_remainder_rows_up():
    plans = plan_segments(
        (
            _segment("s0", 0.5, [(1, 1.0)]),
            _segment("s1", 0.5, [(2, 1.0)]),
        ),
        10,
    )

    # end rows 5 and 10: the trailing partial batch still counts as a step.
    assert boundary_steps(plans, effective_batch=4) == [2, 3]
    assert boundary_steps(plans, effective_batch=32) == [1, 1]


def test_boundary_steps_reject_nonpositive_batch():
    plans = plan_segments((_segment("s0", 1.0, [(1, 1.0)]),), 10)
    with pytest.raises(ValueError, match="effective_batch must be > 0"):
        boundary_steps(plans, effective_batch=0)


def test_segment_index_at_step_maps_steps_to_segments():
    boundaries = [3, 7]

    assert segment_index_at_step(0, boundaries) == 0
    assert segment_index_at_step(3, boundaries) == 0
    assert segment_index_at_step(4, boundaries) == 1
    assert segment_index_at_step(7, boundaries) == 1
    # Steps past the final boundary clamp to the last segment.
    assert segment_index_at_step(99, boundaries) == 1


def test_segment_index_at_step_requires_boundaries():
    with pytest.raises(ValueError, match="at least one boundary"):
        segment_index_at_step(0, [])


def test_replay_fraction_t12():
    assert replay_fraction_t12(((1, 0.5), (2, 0.5))) == 1.0
    assert replay_fraction_t12(((3, 1.0),)) == 0.0
    assert replay_fraction_t12(((1, 1.0), (7, 3.0))) == 0.25
    assert replay_fraction_t12(()) == 0.0
    assert replay_fraction_t12(((1, 0.0), (3, 0.0))) == 0.0
