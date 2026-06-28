"""Tests for scripts/run_pipeline.py orchestration safety checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import STARTING_FEN
from output.stats import PipelineStats


def _valid_example(task: str, tier: int, idx: int = 0) -> dict:
    return {
        "task": task,
        "tier": tier,
        "fen": STARTING_FEN,
        "is_chess960": False,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"FEN: {STARTING_FEN}"},
            {"role": "assistant", "content": f"answer {idx}"},
        ],
        "metadata": {},
    }


def test_run_tier_regenerates_incomplete_existing_output(monkeypatch, tmp_path: Path):
    """A non-empty but underfilled JSONL must not be treated as complete."""
    from scripts import run_pipeline

    class CompleteGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def task_id(self) -> str:
            return "9.1_fake_complete"

        def target_volume(self) -> int:
            return 2

        def generate(self):
            yield _valid_example(self.task_id(), 9, 0)
            yield _valid_example(self.task_id(), 9, 1)

    tier_dir = tmp_path / "tier9"
    tier_dir.mkdir()
    output_path = tier_dir / "9.1_fake_complete.jsonl"
    output_path.write_text(json.dumps(_valid_example("9.1_fake_complete", 9, 0)) + "\n")

    monkeypatch.setattr(run_pipeline, "TIER_OUTPUT_DIR", tmp_path)
    monkeypatch.setitem(run_pipeline.TIER_GENERATORS, 9, [CompleteGenerator])

    run_pipeline.run_tier(9, {}, frozenset(), PipelineStats())

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 2


def test_run_tier_raises_when_generator_underfills(monkeypatch, tmp_path: Path):
    """Fresh generation should fail if a task cannot reach its target."""
    from scripts import run_pipeline

    class UnderfilledGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def task_id(self) -> str:
            return "9.2_fake_underfilled"

        def target_volume(self) -> int:
            return 2

        def generate(self):
            yield _valid_example(self.task_id(), 9, 0)

    monkeypatch.setattr(run_pipeline, "TIER_OUTPUT_DIR", tmp_path)
    monkeypatch.setitem(run_pipeline.TIER_GENERATORS, 9, [UnderfilledGenerator])

    with pytest.raises(RuntimeError, match="underfilled"):
        run_pipeline.run_tier(9, {}, frozenset(), PipelineStats())


def test_run_tier_discards_partial_temp_output_on_generator_error(monkeypatch, tmp_path: Path):
    """A crashing regeneration must not replace the existing JSONL with partial output."""
    from scripts import run_pipeline

    class CrashingGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def task_id(self) -> str:
            return "9.3_fake_crash"

        def target_volume(self) -> int:
            return 2

        def generate(self):
            yield _valid_example(self.task_id(), 9, 99)
            raise RuntimeError("boom")

    tier_dir = tmp_path / "tier9"
    tier_dir.mkdir()
    output_path = tier_dir / "9.3_fake_crash.jsonl"
    old_example = _valid_example("9.3_fake_crash", 9, 0)
    output_path.write_text(json.dumps(old_example) + "\n")

    monkeypatch.setattr(run_pipeline, "TIER_OUTPUT_DIR", tmp_path)
    monkeypatch.setitem(run_pipeline.TIER_GENERATORS, 9, [CrashingGenerator])

    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline.run_tier(9, {}, frozenset(), PipelineStats())

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows == [old_example]
    assert not (tier_dir / "9.3_fake_crash.jsonl.tmp").exists()


def test_run_tier_rejects_generated_rows_for_wrong_task(monkeypatch, tmp_path: Path):
    """A generator must not publish examples labeled as another task."""
    from scripts import run_pipeline

    class WrongTaskGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def task_id(self) -> str:
            return "9.4_expected_task"

        def target_volume(self) -> int:
            return 1

        def generate(self):
            yield _valid_example("9.4_wrong_task", 9, 0)

    monkeypatch.setattr(run_pipeline, "TIER_OUTPUT_DIR", tmp_path)
    monkeypatch.setitem(run_pipeline.TIER_GENERATORS, 9, [WrongTaskGenerator])

    with pytest.raises(RuntimeError, match="validation error"):
        run_pipeline.run_tier(9, {}, frozenset(), PipelineStats())

    assert not (tmp_path / "tier9" / "9.4_expected_task.jsonl").exists()
