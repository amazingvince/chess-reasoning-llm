"""Tests for train.py helper functions — baseline resolution and merging.

These functions are self-contained (only use json, pathlib, logging)
so we can import them directly without triggering torch/transformers.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
from datasets import Dataset

from train import (
    _build_eval_cmd,
    _find_best_historical_baseline,
    _limit_dataset,
    _resolve_run_overrides,
    _update_best_historical_baseline,
)


def _make_args(**overrides) -> Namespace:
    defaults = {
        "smoke_run": False,
        "max_train_examples": None,
        "max_eval_examples": None,
        "max_benchmark_examples_per_split": None,
        "max_steps": None,
        "trainer_eval_steps": None,
        "trainer_save_steps": None,
        "skip_trainer_eval": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


class TestFindBestHistoricalBaseline:
    """Verify best-historical baseline merging."""

    def test_no_prior_phases_returns_none(self, tmp_path: Path):
        result = _find_best_historical_baseline(tmp_path, "a")
        assert result is None

    def test_phase_b_uses_phase_a_results(self, tmp_path: Path):
        phase_a_dir = tmp_path / "phase_a"
        phase_a_dir.mkdir()
        results = {
            "perception": {"board_print": 0.92, "state_tracking": 0.85},
            "rules": {"legal_moves": 0.88},
        }
        with open(phase_a_dir / "eval_predictions.results.json", "w") as fh:
            json.dump(results, fh)

        baseline_path = _find_best_historical_baseline(tmp_path, "b")
        assert baseline_path is not None
        assert baseline_path.exists()

        with open(baseline_path) as fh:
            merged = json.load(fh)
        assert merged["perception"]["board_print"] == 0.92

    def test_phase_c_merges_a_and_b_taking_max(self, tmp_path: Path):
        phase_a_dir = tmp_path / "phase_a"
        phase_a_dir.mkdir()
        with open(phase_a_dir / "eval_predictions.results.json", "w") as fh:
            json.dump({
                "perception": {"board_print": 0.92, "state_tracking": 0.85},
                "rules": {"legal_moves": 0.88},
            }, fh)

        phase_b_dir = tmp_path / "phase_b"
        phase_b_dir.mkdir()
        with open(phase_b_dir / "eval_predictions.results.json", "w") as fh:
            json.dump({
                "perception": {"board_print": 0.90, "state_tracking": 0.87},
                "rules": {"legal_moves": 0.91},
                "evaluation": {"material_balance": 0.96},
            }, fh)

        baseline_path = _find_best_historical_baseline(tmp_path, "c")
        assert baseline_path is not None

        with open(baseline_path) as fh:
            merged = json.load(fh)

        assert merged["perception"]["board_print"] == 0.92    # from A
        assert merged["perception"]["state_tracking"] == 0.87  # from B
        assert merged["rules"]["legal_moves"] == 0.91          # from B
        assert merged["evaluation"]["material_balance"] == 0.96

    def test_phase_bounded_excludes_future_phases(self, tmp_path: Path):
        """Rerunning Phase A after A→B→C should NOT use B/C metrics."""
        for p in ["a", "b", "c"]:
            d = tmp_path / f"phase_{p}"
            d.mkdir()
            with open(d / "eval_predictions.results.json", "w") as fh:
                json.dump({"perception": {"board_print": 0.90 + 0.02 * "abc".index(p)}}, fh)

        # Phase A has no prior phases → should return None
        result = _find_best_historical_baseline(tmp_path, "a")
        assert result is None

        # Phase B should only see A's results
        baseline_path = _find_best_historical_baseline(tmp_path, "b")
        with open(baseline_path) as fh:
            merged = json.load(fh)
        assert merged["perception"]["board_print"] == 0.90  # only from A

        # Phase C should see A and B
        baseline_path = _find_best_historical_baseline(tmp_path, "c")
        with open(baseline_path) as fh:
            merged = json.load(fh)
        assert merged["perception"]["board_print"] == 0.92  # max(A=0.90, B=0.92)

    def test_stale_global_baseline_ignored(self, tmp_path: Path):
        """A leftover best_baseline.json should NOT be reused blindly."""
        # Create a stale global baseline with inflated metrics
        best_path = tmp_path / "best_baseline.json"
        with open(best_path, "w") as fh:
            json.dump({"perception": {"board_print": 0.99}}, fh)

        # Create actual phase A results
        phase_a_dir = tmp_path / "phase_a"
        phase_a_dir.mkdir()
        with open(phase_a_dir / "eval_predictions.results.json", "w") as fh:
            json.dump({"perception": {"board_print": 0.92}}, fh)

        # Phase B should build from phase A results, not the stale global
        baseline_path = _find_best_historical_baseline(tmp_path, "b")
        with open(baseline_path) as fh:
            merged = json.load(fh)
        assert merged["perception"]["board_print"] == 0.92  # from A, not 0.99

    def test_acpl_takes_min_not_max(self, tmp_path: Path):
        phase_a_dir = tmp_path / "phase_a"
        phase_a_dir.mkdir()
        with open(phase_a_dir / "eval_predictions.results.json", "w") as fh:
            json.dump({"planning": {"acpl": 300.0, "puzzle_solve": 0.15}}, fh)

        phase_b_dir = tmp_path / "phase_b"
        phase_b_dir.mkdir()
        with open(phase_b_dir / "eval_predictions.results.json", "w") as fh:
            json.dump({"planning": {"acpl": 180.0, "puzzle_solve": 0.22}}, fh)

        baseline_path = _find_best_historical_baseline(tmp_path, "c")
        with open(baseline_path) as fh:
            merged = json.load(fh)

        assert merged["planning"]["acpl"] == 180.0
        assert merged["planning"]["puzzle_solve"] == 0.22


class TestUpdateBestHistoricalBaseline:
    """_update_best_historical_baseline is now a no-op stub."""

    def test_noop(self, tmp_path: Path):
        """Should not create any files."""
        phase_a_dir = tmp_path / "phase_a"
        phase_a_dir.mkdir()
        with open(phase_a_dir / "eval_predictions.results.json", "w") as fh:
            json.dump({"perception": {"board_print": 0.92}}, fh)

        _update_best_historical_baseline(tmp_path, "a")

        # Should NOT create best_baseline.json (baselines are built on demand)
        assert not (tmp_path / "best_baseline.json").exists()


class TestRunOverrides:
    def test_full_run_defaults_bound_trainer_eval(self):
        overrides = _resolve_run_overrides(_make_args())
        assert overrides.max_eval_examples == 2048
        assert overrides.eval_steps is None
        assert overrides.save_steps is None
        assert overrides.skip_trainer_eval is False

    def test_zero_eval_limit_means_full_eval_set(self):
        overrides = _resolve_run_overrides(_make_args(max_eval_examples=0))
        assert overrides.max_eval_examples is None

    def test_smoke_defaults_are_applied(self):
        overrides = _resolve_run_overrides(_make_args(smoke_run=True))
        assert overrides.max_train_examples == 128
        assert overrides.max_eval_examples == 64
        assert overrides.max_benchmark_examples_per_split == 32
        assert overrides.max_steps == 10
        assert overrides.eval_steps == 5
        assert overrides.save_steps == 5
        assert overrides.logging_steps == 1

    def test_explicit_values_override_smoke_defaults(self):
        overrides = _resolve_run_overrides(
            _make_args(
                smoke_run=True,
                max_train_examples=20,
                max_eval_examples=10,
                max_benchmark_examples_per_split=8,
                max_steps=6,
            ),
        )
        assert overrides.max_train_examples == 20
        assert overrides.max_eval_examples == 10
        assert overrides.max_benchmark_examples_per_split == 8
        assert overrides.max_steps == 6
        assert overrides.eval_steps == 3
        assert overrides.save_steps == 3
        assert overrides.logging_steps == 1

    def test_explicit_trainer_cadence_is_preserved(self):
        overrides = _resolve_run_overrides(
            _make_args(trainer_eval_steps=1000, trainer_save_steps=2000),
        )
        assert overrides.eval_steps == 1000
        assert overrides.save_steps == 2000

    def test_save_only_cadence_sets_eval_to_match(self):
        overrides = _resolve_run_overrides(_make_args(trainer_save_steps=2000))
        assert overrides.eval_steps == 2000
        assert overrides.save_steps == 2000

    def test_rejects_incompatible_trainer_cadence(self):
        with pytest.raises(ValueError, match="multiple"):
            _resolve_run_overrides(
                _make_args(trainer_eval_steps=3000, trainer_save_steps=2000),
            )

    def test_skip_trainer_eval_is_preserved(self):
        overrides = _resolve_run_overrides(_make_args(skip_trainer_eval=True))
        assert overrides.skip_trainer_eval is True


class TestDatasetLimit:
    def test_limit_dataset_caps_examples(self):
        ds = Dataset.from_dict({"value": list(range(10))})
        limited = _limit_dataset(ds, 4, seed=123)
        assert len(limited) == 4

    def test_limit_dataset_is_noop_when_under_limit(self):
        ds = Dataset.from_dict({"value": [1, 2, 3]})
        limited = _limit_dataset(ds, 10)
        assert len(limited) == 3
        assert limited["value"] == [1, 2, 3]


class TestEvalCommand:
    def test_build_eval_cmd_includes_backend_and_limits(self, tmp_path: Path):
        cmd = _build_eval_cmd(
            "model-id",
            tmp_path / "benchmark",
            tmp_path / "predictions.jsonl",
            baseline_path=tmp_path / "baseline.json",
            phase="c",
            pass_k=8,
            stockfish_path="C:/stockfish.exe",
            inference_backend="vllm",
            attn_implementation="sdpa",
            eval_batch_size=32,
            eval_max_new_tokens=128,
            eval_acpl_depth=12,
            no_acpl=True,
            full_acpl_report=True,
            report_only=True,
            soft_gate=True,
            max_examples_per_split=32,
            wandb_project="proj",
            wandb_run_name="phase-c-post-eval",
            wandb_group="curriculum-1",
        )
        assert "--inference-backend" in cmd
        assert "vllm" in cmd
        assert "--attn-implementation" in cmd
        assert "sdpa" in cmd
        assert "--batch-size" in cmd
        assert "128" in cmd
        assert "--no-acpl" in cmd
        assert "--full-acpl-report" in cmd
        assert "--report-only" in cmd
        assert "--soft-gate" in cmd
        assert "--max-examples-per-split" in cmd
        assert "32" in cmd
        assert "--pass-k" in cmd
        assert "--stockfish-path" in cmd
        assert "--wandb-project" in cmd
        assert "proj" in cmd
        assert "--wandb-run-name" in cmd
        assert "phase-c-post-eval" in cmd
        assert "--wandb-group" in cmd
