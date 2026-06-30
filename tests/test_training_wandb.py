from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path


def test_wandb_config_error_requires_auth_for_enabled_cloud_logging(tmp_path: Path) -> None:
    from chess_llm.training.wandb_utils import wandb_config_error

    assert wandb_config_error(
        no_wandb=False,
        env={},
        netrc_path=tmp_path / "missing-netrc",
    )


def test_wandb_config_error_accepts_api_key_env(tmp_path: Path) -> None:
    from chess_llm.training.wandb_utils import wandb_config_error

    assert wandb_config_error(
        no_wandb=False,
        env={"WANDB_API_KEY": "secret"},
        netrc_path=tmp_path / "missing-netrc",
    ) is None


def test_wandb_config_error_accepts_saved_netrc_login(tmp_path: Path) -> None:
    from chess_llm.training.wandb_utils import wandb_config_error

    netrc_path = tmp_path / ".netrc"
    netrc_path.write_text(
        "machine api.wandb.ai login user password saved-key\n",
        encoding="utf-8",
    )

    assert wandb_config_error(
        no_wandb=False,
        env={},
        netrc_path=netrc_path,
    ) is None


def test_wandb_config_error_allows_explicit_no_wandb(tmp_path: Path) -> None:
    from chess_llm.training.wandb_utils import wandb_config_error

    assert wandb_config_error(
        no_wandb=True,
        env={},
        netrc_path=tmp_path / "missing-netrc",
    ) is None


def test_wandb_config_error_allows_explicit_offline_mode(tmp_path: Path) -> None:
    from chess_llm.training.wandb_utils import wandb_config_error

    assert wandb_config_error(
        no_wandb=False,
        env={"WANDB_MODE": "offline"},
        netrc_path=tmp_path / "missing-netrc",
    ) is None


def test_wandb_config_error_rejects_offline_when_online_required(tmp_path: Path) -> None:
    from chess_llm.training.wandb_utils import wandb_config_error

    error = wandb_config_error(
        no_wandb=False,
        env={"WANDB_MODE": "offline"},
        netrc_path=tmp_path / "missing-netrc",
        allow_offline=False,
    )

    assert error is not None
    assert "WANDB_MODE=offline" in error
    assert "--allow-wandb-offline" in error


def test_eval_returns_infra_failure_when_wandb_auth_is_missing(monkeypatch, tmp_path: Path) -> None:
    from chess_llm.evals.benchmark import BenchmarkExample
    from chess_llm.training import evaluate
    from chess_llm.training.eval_exit_codes import EVAL_INFRA_FAILURE_EXIT_CODE

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "planning.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        evaluate,
        "load_benchmark",
        lambda _path, **_kwargs: [
            BenchmarkExample(
                example_id="planning_00000",
                split="planning",
                task_type="best_move",
                fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                prompt="FEN: ...",
                gold_answer="e2e4",
                metric_type="move_extraction",
                metadata={},
            )
        ],
    )
    monkeypatch.setattr(
        evaluate,
        "load_model_and_tokenizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model should not load when W&B auth is missing")
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "wandb_config_error",
        lambda **_kwargs: "W&B logging is enabled but no API key is configured.",
    )

    result = evaluate.run_evaluation(
        evaluate.EvaluationConfig(
            model="model-id",
            benchmark_dir=benchmark_dir,
            output=tmp_path / "predictions.jsonl",
            no_acpl=True,
            no_wandb=False,
        )
    )

    assert result.return_code == EVAL_INFRA_FAILURE_EXIT_CODE
    eval_run = result.eval_run_path.read_text(encoding="utf-8")
    assert "wandb_auth_missing" in eval_run


def test_eval_rejects_wandb_offline_mode_by_default(monkeypatch, tmp_path: Path) -> None:
    from chess_llm.evals.benchmark import BenchmarkExample
    from chess_llm.training import evaluate
    from chess_llm.training.eval_exit_codes import EVAL_INFRA_FAILURE_EXIT_CODE

    monkeypatch.setenv("WANDB_MODE", "offline")
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "planning.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        evaluate,
        "load_benchmark",
        lambda _path, **_kwargs: [
            BenchmarkExample(
                example_id="planning_00000",
                split="planning",
                task_type="best_move",
                fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                prompt="FEN: ...",
                gold_answer="e2e4",
                metric_type="move_extraction",
                metadata={},
            )
        ],
    )
    monkeypatch.setattr(
        evaluate,
        "load_model_and_tokenizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model should not load when W&B is accidentally offline")
        ),
    )

    result = evaluate.run_evaluation(
        evaluate.EvaluationConfig(
            model="model-id",
            benchmark_dir=benchmark_dir,
            output=tmp_path / "predictions.jsonl",
            no_acpl=True,
            no_wandb=False,
        )
    )

    assert result.return_code == EVAL_INFRA_FAILURE_EXIT_CODE
    eval_run = result.eval_run_path.read_text(encoding="utf-8")
    assert "wandb_offline_disallowed" in eval_run
    assert "WANDB_MODE=offline" in eval_run


def test_eval_classifies_missing_wandb_auth_separately(monkeypatch, tmp_path: Path) -> None:
    from chess_llm.evals.benchmark import BenchmarkExample
    from chess_llm.training import evaluate
    from chess_llm.training.eval_exit_codes import EVAL_INFRA_FAILURE_EXIT_CODE

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "planning.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(
        evaluate,
        "load_benchmark",
        lambda _path, **_kwargs: [
            BenchmarkExample(
                example_id="planning_00000",
                split="planning",
                task_type="best_move",
                fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                prompt="FEN: ...",
                gold_answer="e2e4",
                metric_type="move_extraction",
                metadata={},
            )
        ],
    )
    monkeypatch.setattr(
        evaluate,
        "load_model_and_tokenizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model should not load when W&B auth is missing")
        ),
    )

    result = evaluate.run_evaluation(
        evaluate.EvaluationConfig(
            model="model-id",
            benchmark_dir=benchmark_dir,
            output=tmp_path / "predictions.jsonl",
            no_acpl=True,
            no_wandb=False,
        )
    )

    assert result.return_code == EVAL_INFRA_FAILURE_EXIT_CODE
    eval_run = result.eval_run_path.read_text(encoding="utf-8")
    assert "wandb_auth_missing" in eval_run
    assert "wandb_offline_disallowed" not in eval_run


def test_train_returns_infra_failure_when_wandb_auth_is_missing(monkeypatch, tmp_path: Path) -> None:
    from chess_llm.training import train
    from chess_llm.training.eval_exit_codes import EVAL_INFRA_FAILURE_EXIT_CODE

    args = Namespace(
        phase="a",
        data_root=tmp_path / "data",
        output_root=tmp_path / "checkpoints",
        benchmark_dir=tmp_path / "benchmark",
        base_model=None,
        dry_run=False,
        smoke_run=False,
        eval_only=False,
        skip_eval=False,
        require_phase_gate=False,
        max_train_examples=None,
        task_upsample=[],
        max_eval_examples=None,
        max_benchmark_examples_per_split=None,
        full_benchmark_eval=False,
        max_steps=None,
        trainer_eval_steps=None,
        trainer_save_steps=None,
        skip_trainer_eval=False,
        wandb_project="chess-sft",
        no_wandb=False,
        allow_wandb_offline=False,
        run_name="missing-wandb",
        wandb_group=None,
        disable_liger_kernel=False,
        inference_backend="transformers",
        attn_implementation="auto",
        eval_batch_size=1,
        eval_max_new_tokens=16,
        eval_acpl_depth=1,
        no_acpl=True,
        full_acpl_report=False,
        stockfish_path="stockfish",
    )

    monkeypatch.setattr(train, "parse_args", lambda: args)
    monkeypatch.setattr(
        train,
        "wandb_config_error",
        lambda **_kwargs: "W&B logging is enabled but no API key is configured.",
    )

    assert train.main() == EVAL_INFRA_FAILURE_EXIT_CODE


def test_train_sets_wandb_project_and_clears_stale_group_env(monkeypatch):
    from chess_llm.training import train

    monkeypatch.setenv("WANDB_PROJECT", "stale-project")
    monkeypatch.setenv("WANDB_GROUP", "stale-group")
    monkeypatch.setenv("WANDB_RUN_GROUP", "stale-run-group")

    train._configure_wandb_env(
        no_wandb=False,
        wandb_project="chess-sft",
        wandb_group=None,
    )

    assert os.environ["WANDB_PROJECT"] == "chess-sft"
    assert "WANDB_GROUP" not in os.environ
    assert "WANDB_RUN_GROUP" not in os.environ


def test_train_sets_wandb_group_env(monkeypatch):
    from chess_llm.training import train

    monkeypatch.delenv("WANDB_GROUP", raising=False)
    monkeypatch.delenv("WANDB_RUN_GROUP", raising=False)

    train._configure_wandb_env(
        no_wandb=False,
        wandb_project="chess-sft",
        wandb_group="phase-a",
    )

    assert os.environ["WANDB_PROJECT"] == "chess-sft"
    assert os.environ["WANDB_GROUP"] == "phase-a"
    assert os.environ["WANDB_RUN_GROUP"] == "phase-a"


def test_train_allows_wandb_offline_for_smoke_runs(monkeypatch, tmp_path: Path) -> None:
    from chess_llm.training import train

    args = Namespace(
        phase="a",
        data_root=tmp_path / "data",
        output_root=tmp_path / "checkpoints",
        benchmark_dir=tmp_path / "benchmark",
        base_model=None,
        dry_run=False,
        smoke_run=True,
        eval_only=False,
        skip_eval=False,
        require_phase_gate=False,
        max_train_examples=None,
        task_upsample=[],
        max_eval_examples=None,
        max_benchmark_examples_per_split=None,
        full_benchmark_eval=False,
        max_steps=None,
        trainer_eval_steps=None,
        trainer_save_steps=None,
        skip_trainer_eval=False,
        wandb_project="chess-sft",
        no_wandb=False,
        allow_wandb_offline=False,
        run_name="offline-smoke",
        wandb_group=None,
        disable_liger_kernel=False,
        inference_backend="transformers",
        attn_implementation="auto",
        eval_batch_size=1,
        eval_max_new_tokens=16,
        eval_acpl_depth=1,
        no_acpl=True,
        full_acpl_report=False,
        stockfish_path="stockfish",
    )
    captured = {}

    monkeypatch.setattr(train, "parse_args", lambda: args)
    monkeypatch.setattr(
        train,
        "wandb_config_error",
        lambda **kwargs: captured.setdefault("kwargs", kwargs) and "stop-here",
    )

    train.main()

    assert captured["kwargs"]["allow_offline"] is True
