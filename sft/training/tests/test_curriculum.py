from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from run_curriculum import (
    _build_pre_eval_cmd,
    _build_train_phase_cmd,
    _phase_sequence,
    _resolve_phase_start_model,
)


def _make_args(tmp_path: Path, **overrides) -> Namespace:
    defaults = {
        "data_root": tmp_path / "data",
        "output_root": tmp_path / "checkpoints",
        "benchmark_dir": tmp_path / "benchmark",
        "smoke_run": False,
        "pre_eval": False,
        "skip_pre_eval": False,
        "max_train_examples": None,
        "max_eval_examples": None,
        "max_benchmark_examples_per_split": None,
        "max_steps": None,
        "trainer_eval_steps": None,
        "trainer_save_steps": None,
        "skip_trainer_eval": False,
        "require_phase_gate": False,
        "inference_backend": "transformers",
        "attn_implementation": "auto",
        "eval_batch_size": 16,
        "eval_max_new_tokens": 256,
        "eval_acpl_depth": 20,
        "no_acpl": False,
        "full_acpl_report": False,
        "stockfish_path": None,
        "wandb_project": "chess-sft",
        "wandb_group": None,
        "run_prefix": None,
        "no_wandb": False,
        "dry_run": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


class TestPhaseSequence:
    def test_includes_bounds(self):
        assert _phase_sequence("a", "c") == ["a", "b", "c"]
        assert _phase_sequence("b", "c") == ["b", "c"]
        assert _phase_sequence("c", "c") == ["c"]

    def test_rejects_reverse_order(self):
        with pytest.raises(ValueError):
            _phase_sequence("c", "a")


class TestCurriculumCommands:
    def test_resolve_phase_start_model_simulates_prior_dry_run_output(self, tmp_path: Path):
        output_root = tmp_path / "checkpoints"
        model_path = _resolve_phase_start_model(
            "b",
            output_root,
            dry_run=True,
            planned_completed_phases={"a"},
        )
        assert model_path == str(output_root / "phase_a" / "best")

    def test_build_train_phase_cmd_passes_wandb_and_limits(self, tmp_path: Path):
        args = _make_args(
            tmp_path,
            smoke_run=True,
            max_train_examples=128,
            max_eval_examples=64,
            max_benchmark_examples_per_split=32,
            max_steps=10,
            require_phase_gate=True,
            stockfish_path="/usr/games/stockfish",
            wandb_group="group-1",
        )
        cmd = _build_train_phase_cmd(
            "b",
            args,
            curriculum_id="curriculum-test",
            wandb_group="group-1",
        )

        assert "--phase" in cmd
        assert "b" in cmd
        assert "--smoke-run" in cmd
        assert "--max-train-examples" in cmd
        assert "--max-eval-examples" in cmd
        assert "--max-benchmark-examples-per-split" in cmd
        assert "--max-steps" in cmd
        assert "--require-phase-gate" in cmd
        assert "--attn-implementation" in cmd
        assert "--eval-batch-size" in cmd
        assert "--eval-max-new-tokens" in cmd
        assert "--eval-acpl-depth" in cmd
        assert "--wandb-project" in cmd
        assert "--wandb-group" in cmd
        assert "curriculum-test-phase-b-train" in cmd

    def test_build_pre_eval_cmd_is_lightweight_report_only(self, tmp_path: Path):
        args = _make_args(tmp_path, max_benchmark_examples_per_split=24)
        phase_a_dir = args.output_root / "phase_a"
        phase_a_dir.mkdir(parents=True)
        with open(phase_a_dir / "eval_predictions.results.json", "w", encoding="utf-8") as fh:
            json.dump({"perception": {"board_print": 0.91}}, fh)

        cmd = _build_pre_eval_cmd(
            "b",
            "model-id",
            args,
            curriculum_id="curriculum-test",
            wandb_group="group-1",
        )

        assert "--baseline" not in cmd
        assert "--phase" not in cmd
        assert "--pass-k" not in cmd
        assert "--no-acpl" in cmd
        assert "--report-only" in cmd
        assert "--output" in cmd
        assert str(args.output_root / "phase_b" / "pre_eval_predictions.jsonl") in cmd
        assert "--wandb-run-name" in cmd
        assert "curriculum-test-phase-b-pre-eval" in cmd

    def test_build_pre_eval_cmd_can_disable_wandb(self, tmp_path: Path):
        args = _make_args(tmp_path, no_wandb=True)
        cmd = _build_pre_eval_cmd(
            "a",
            "base-model",
            args,
            curriculum_id="curriculum-test",
            wandb_group=None,
        )

        assert "--no-wandb" in cmd
        assert "--wandb-project" not in cmd
