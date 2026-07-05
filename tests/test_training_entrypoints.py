from __future__ import annotations

import builtins
import importlib
import json
import logging
from argparse import Namespace
from pathlib import Path
import sys
import types
import warnings


def _expire_module(monkeypatch, module_name):
    """Force *module_name* to re-import during this test only.

    The original module object is restored at teardown both in sys.modules and
    as its parent package's attribute, so later tests never see a
    stub-flavored or freshly re-imported module. Re-importing torch from
    scratch in-process would also break its C extension loading on Windows.
    """
    original = sys.modules.get(module_name)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    if original is not None and "." in module_name:
        parent_name, attr = module_name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            monkeypatch.setattr(parent, attr, original, raising=False)


def _expire_training_args(monkeypatch):
    """Drop the cached training_args module so it re-imports under this test's
    stubbed trl."""
    _expire_module(monkeypatch, "chess_llm.training.training_args")


def test_training_entrypoint_modules_import_without_heavy_gpu_deps(monkeypatch):
    for module_name in [
        "chess_llm.training.evaluate",
        "chess_llm.training.run_curriculum",
        "chess_llm.training.train",
        "torch",
        "transformers",
        "tqdm",
        "trl",
        "vllm",
    ]:
        _expire_module(monkeypatch, module_name)

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"torch", "transformers", "tqdm", "trl", "vllm"}:
            raise AssertionError(f"heavy import at training entrypoint import time: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert importlib.import_module("chess_llm.training.train")
    assert importlib.import_module("chess_llm.training.evaluate")
    assert importlib.import_module("chess_llm.training.run_curriculum")


def test_package_train_eval_command_uses_module_entrypoint(tmp_path: Path):
    from chess_llm.training.train import _build_eval_cmd

    cmd = _build_eval_cmd(
        "model-id",
        tmp_path / "benchmark",
        tmp_path / "predictions.jsonl",
    )

    assert cmd[:3] == [sys.executable, "-m", "chess_llm.training.evaluate"]


def test_train_cli_accepts_base_model_override(monkeypatch):
    from chess_llm.training import train

    monkeypatch.setattr(
        sys,
        "argv",
        ["chess-llm-train", "--phase", "a", "--base-model", "Qwen/Qwen3.5-0.8B"],
    )

    args = train.parse_args()

    assert args.base_model == "Qwen/Qwen3.5-0.8B"


def test_train_registers_pinned_flash_attention_variant(monkeypatch):
    from chess_llm.training import train

    monkeypatch.setitem(
        sys.modules,
        "trl",
        sys.modules.get("trl")
        or types.SimpleNamespace(SFTConfig=type("FakeSFTConfig", (), {})),
    )
    fake_sft_trainer = types.SimpleNamespace(FLASH_ATTENTION_VARIANTS=set())

    def fake_import_module(name: str):
        if name == "trl.trainer.sft_trainer":
            return fake_sft_trainer
        raise AssertionError(name)

    monkeypatch.setattr(train.importlib, "import_module", fake_import_module)

    train._register_trl_flash_attention_variant("kernels-community/flash-attn2@abc")

    assert fake_sft_trainer.FLASH_ATTENTION_VARIANTS == {
        "kernels-community/flash-attn2@abc",
    }


def test_train_does_not_register_non_flash_attention_variants(monkeypatch):
    from chess_llm.training import train

    monkeypatch.setitem(
        sys.modules,
        "trl",
        sys.modules.get("trl")
        or types.SimpleNamespace(SFTConfig=type("FakeSFTConfig", (), {})),
    )
    fake_sft_trainer = types.SimpleNamespace(FLASH_ATTENTION_VARIANTS=set())
    monkeypatch.setattr(
        train.importlib,
        "import_module",
        lambda name: fake_sft_trainer,
    )

    # sdpa/eager must stay unregistered so TRL's packing contamination
    # guard stays loud for non-flash backends.
    for attn_implementation in ["sdpa", "eager", None]:
        train._register_trl_flash_attention_variant(attn_implementation)

    assert fake_sft_trainer.FLASH_ATTENTION_VARIANTS == set()


def test_training_args_identify_genuine_flash_attention_backends(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "trl",
        sys.modules.get("trl")
        or types.SimpleNamespace(SFTConfig=type("FakeSFTConfig", (), {})),
    )
    from chess_llm.training.training_args import is_flash_attention_implementation

    assert is_flash_attention_implementation("flash_attention_2") is True
    assert is_flash_attention_implementation("flash_attention_3") is True
    assert is_flash_attention_implementation("kernels-community/flash-attn2@abc") is True
    assert is_flash_attention_implementation("sdpa") is False
    assert is_flash_attention_implementation("eager") is False
    assert is_flash_attention_implementation(None) is False


def test_train_metrics_adds_token_throughput_from_trainer_state():
    from chess_llm.training.train import _augment_train_metrics_with_token_throughput

    trainer = types.SimpleNamespace(
        state=types.SimpleNamespace(num_input_tokens_seen=271200),
    )

    metrics = _augment_train_metrics_with_token_throughput(
        trainer,
        {"train_runtime": 33.76, "train_loss": 0.5},
    )

    assert metrics["train_num_tokens"] == 271200
    assert metrics["train_tokens_per_second"] == 271200 / 33.76
    assert metrics["train_loss"] == 0.5


def test_train_metrics_adds_token_throughput_from_log_history():
    from chess_llm.training.train import _augment_train_metrics_with_token_throughput

    trainer = types.SimpleNamespace(
        state=types.SimpleNamespace(
            num_input_tokens_seen=None,
            log_history=[
                {"num_tokens": "100"},
                {"loss": 0.1},
                {"num_tokens": "2800"},
            ],
        ),
    )

    metrics = _augment_train_metrics_with_token_throughput(
        trainer,
        {"train_runtime": "2.0"},
    )

    assert metrics["train_num_tokens"] == 2800
    assert metrics["train_tokens_per_second"] == 1400


def test_train_metrics_skip_token_throughput_when_inputs_missing():
    from chess_llm.training.train import _augment_train_metrics_with_token_throughput

    trainer = types.SimpleNamespace(state=types.SimpleNamespace(log_history=[]))

    assert _augment_train_metrics_with_token_throughput(
        trainer,
        {"train_runtime": 0},
    ) == {"train_runtime": 0}


def test_train_cli_parses_task_upsample_overrides(monkeypatch):
    from chess_llm.training import train

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chess-llm-train",
            "--phase",
            "a",
            "--task-upsample",
            "1.5_state_tracking=8",
            "--task-upsample",
            "2.3_move_legality_check=2",
        ],
    )

    args = train.parse_args()
    parsed = train._parse_task_upsample_overrides(args.task_upsample)

    assert parsed == {
        "1.5_state_tracking": 8,
        "2.3_move_legality_check": 2,
    }


def test_train_cli_accepts_data_transform_overrides(monkeypatch):
    from chess_llm.training import train

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chess-llm-train",
            "--phase",
            "c",
            "--task-include",
            "7.1_best_move_selection",
            "--task-exclude",
            "7.8_candidate_ratings",
            "--move-only-task",
            "7.1_best_move_selection",
        ],
    )

    args = train.parse_args()
    config = train._build_data_transform_config(args)

    assert args.task_include == ["7.1_best_move_selection"]
    assert args.task_exclude == ["7.8_candidate_ratings"]
    assert args.move_only_task == ["7.1_best_move_selection"]
    assert config.task_include == frozenset({"7.1_best_move_selection"})
    assert config.task_exclude == frozenset({"7.8_candidate_ratings"})
    assert config.move_only_tasks == frozenset({"7.1_best_move_selection"})


def test_train_cli_accepts_num_train_epochs_override(monkeypatch):
    from chess_llm.training import train

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chess-llm-train",
            "--phase",
            "a",
            "--num-train-epochs",
            "1",
        ],
    )

    args = train.parse_args()

    assert args.num_train_epochs == 1.0


def test_train_cli_accepts_learning_rate_override(monkeypatch):
    from chess_llm.training import train

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chess-llm-train",
            "--phase",
            "a",
            "--learning-rate",
            "2e-6",
        ],
    )

    args = train.parse_args()

    assert args.learning_rate == 2e-6


def test_train_cli_accepts_packing_and_max_length_overrides(monkeypatch):
    from chess_llm.training import train

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chess-llm-train",
            "--phase",
            "a",
            "--packing",
            "on",
            "--max-length",
            "1024",
        ],
    )

    args = train.parse_args()

    assert args.packing == "on"
    assert args.max_length == 1024


def test_train_cli_accepts_auto_resume_checkpoint(monkeypatch):
    from chess_llm.training import train

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chess-llm-train",
            "--phase",
            "a",
            "--resume-from-checkpoint",
        ],
    )

    args = train.parse_args()

    assert args.resume_from_checkpoint == "auto"


def test_dry_run_training_details_reports_rehearsal_schedule(monkeypatch, tmp_path: Path):
    monkeypatch.setitem(
        sys.modules,
        "trl",
        sys.modules.get("trl")
        or types.SimpleNamespace(SFTConfig=type("FakeSFTConfig", (), {})),
    )
    _expire_training_args(monkeypatch)

    from chess_llm.training import train
    from chess_llm.training.phases import PHASE_A

    monkeypatch.setenv("WANDB_GIT_COMMIT", "abc123")
    args = Namespace(
        no_wandb=False,
        wandb_project="chess-sft",
        wandb_group="phase-a-t12-v25000",
        run_name="phase-a-t12-v25000-train",
        inference_backend="transformers",
        skip_eval=True,
        attn_implementation="auto",
        resume_from_checkpoint=None,
        output_root=tmp_path / "checkpoints",
    )
    overrides = train.RunOverrides(
        num_train_epochs=1,
        save_steps=1000,
        skip_trainer_eval=True,
    )

    details = train._build_dry_run_training_details(
        PHASE_A,
        args,
        overrides,
        train_dataset_size=725000,
    )
    assert "Estimated optimizer steps: 22657" in details
    assert "Warmup steps: 680 (ratio 0.03)" in details
    assert "Trainer eval: disabled" in details
    assert "Checkpoint saves: every 1000 steps, keep 3 (resumable)" in details
    assert "W&B: enabled project=chess-sft group=phase-a-t12-v25000 run=phase-a-t12-v25000-train git=abc123" in details
    assert "Post-training benchmark eval: skipped" in details
    assert "Resume: disabled" in details


def test_train_dry_run_estimates_steps_from_actual_train_split(monkeypatch, tmp_path: Path):
    from chess_llm.training import train
    from chess_llm.training.data import mixer

    class FakeDataset:
        def __init__(self, size: int):
            self.size = size

        def __len__(self):
            return self.size

    captured: dict[str, int] = {}

    monkeypatch.setattr(
        mixer,
        "summarize_phase_data",
        lambda *args, **kwargs: {"tier_1": 100, "total": 100},
    )
    monkeypatch.setattr(
        mixer,
        "build_phase_dataset",
        lambda *args, **kwargs: (FakeDataset(90), FakeDataset(10)),
    )

    def fake_details(*args, train_dataset_size: int, **kwargs):
        captured["train_dataset_size"] = train_dataset_size
        return []

    monkeypatch.setattr(train, "_build_dry_run_training_details", fake_details)
    monkeypatch.setattr(
        train,
        "parse_args",
        lambda: Namespace(
            phase="a",
            data_root=tmp_path / "data",
            output_root=tmp_path / "checkpoints",
            benchmark_dir=tmp_path / "benchmark",
            base_model=None,
            dry_run=True,
            smoke_run=False,
            eval_only=False,
            skip_eval=True,
            require_phase_gate=False,
            max_train_examples=None,
            task_upsample=[],
            max_eval_examples=None,
            max_benchmark_examples_per_split=None,
            full_benchmark_eval=False,
            max_steps=None,
            num_train_epochs=1,
            trainer_eval_steps=None,
            trainer_save_steps=1000,
            skip_trainer_eval=True,
            resume_from_checkpoint=None,
            wandb_project="chess-sft",
            no_wandb=True,
            allow_wandb_offline=False,
            run_name=None,
            wandb_group=None,
            inference_backend="transformers",
            attn_implementation="auto",
        ),
    )

    assert train.main() == 0
    assert captured["train_dataset_size"] == 90


def test_train_dry_run_resolves_attn_like_real_run_for_packing_report(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setitem(
        sys.modules,
        "trl",
        sys.modules.get("trl")
        or types.SimpleNamespace(SFTConfig=type("FakeSFTConfig", (), {})),
    )
    import chess_llm.training.training_args as training_args_module
    from chess_llm.training import train
    from chess_llm.training.data import mixer

    class FakeDataset:
        def __len__(self):
            return 10

    monkeypatch.setattr(
        mixer,
        "summarize_phase_data",
        lambda *args, **kwargs: {"total": 10},
    )
    monkeypatch.setattr(
        mixer,
        "build_phase_dataset",
        lambda *args, **kwargs: (FakeDataset(), FakeDataset()),
    )
    monkeypatch.setattr(
        train,
        "_build_dry_run_training_details",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        train,
        "attention_candidates",
        lambda requested: ["flash_attention_2", "sdpa", None],
    )

    captured: dict[str, object] = {}

    def fake_resolve_packing_settings(attn_implementation, packing="auto"):
        captured["attn"] = attn_implementation
        captured["packing"] = packing
        return True, True

    monkeypatch.setattr(
        training_args_module,
        "resolve_packing_settings",
        fake_resolve_packing_settings,
    )
    monkeypatch.setattr(
        train,
        "parse_args",
        lambda: Namespace(
            phase="a",
            data_root=tmp_path / "data",
            output_root=tmp_path / "checkpoints",
            benchmark_dir=tmp_path / "benchmark",
            base_model=None,
            dry_run=True,
            smoke_run=False,
            eval_only=False,
            skip_eval=True,
            require_phase_gate=False,
            max_train_examples=None,
            task_upsample=[],
            max_eval_examples=None,
            max_benchmark_examples_per_split=None,
            full_benchmark_eval=False,
            max_steps=None,
            num_train_epochs=None,
            trainer_eval_steps=None,
            trainer_save_steps=None,
            skip_trainer_eval=False,
            resume_from_checkpoint=None,
            wandb_project="chess-sft",
            no_wandb=True,
            allow_wandb_offline=False,
            run_name=None,
            wandb_group=None,
            inference_backend="transformers",
            attn_implementation="auto",
            packing="auto",
            max_length=2048,
        ),
    )

    assert train.main() == 0
    # The dry run must report packing for the backend the real run would
    # select, not for a literal attn=None under --attn-implementation auto.
    assert captured["attn"] == "flash_attention_2"
    assert captured["packing"] == "auto"


def _eval_only_args(tmp_path: Path) -> Namespace:
    return Namespace(
        phase="a",
        data_root=tmp_path / "data",
        output_root=tmp_path / "checkpoints",
        benchmark_dir=tmp_path / "benchmark",
        base_model=None,
        dry_run=False,
        smoke_run=False,
        eval_only=True,
        skip_eval=False,
        require_phase_gate=False,
        max_train_examples=None,
        task_upsample=[],
        max_eval_examples=None,
        max_benchmark_examples_per_split=None,
        full_benchmark_eval=False,
        max_steps=None,
        num_train_epochs=None,
        trainer_eval_steps=None,
        trainer_save_steps=None,
        skip_trainer_eval=False,
        resume_from_checkpoint=None,
        wandb_project="chess-sft",
        no_wandb=True,
        allow_wandb_offline=False,
        run_name=None,
        wandb_group=None,
        inference_backend="transformers",
        attn_implementation="auto",
        eval_batch_size=16,
        eval_max_new_tokens=256,
        eval_acpl_depth=20,
        no_acpl=True,
        full_acpl_report=False,
        stockfish_path=None,
    )


def test_eval_only_on_untrained_phase_writes_separate_artifact(monkeypatch, tmp_path: Path):
    from chess_llm.training import train

    captured: dict[str, object] = {}

    def fake_run_eval(model_path, benchmark_dir, pred_path, **kwargs):
        captured["model"] = model_path
        captured["pred_path"] = pred_path
        return 0

    monkeypatch.setattr(train, "_run_eval", fake_run_eval)
    monkeypatch.setattr(train, "parse_args", lambda: _eval_only_args(tmp_path))

    assert train.main() == 0
    # No best/ checkpoint: fallback-model metrics must not land in
    # eval_predictions.results.json, which later merges as this phase's
    # historical baseline.
    assert captured["pred_path"] == (
        tmp_path / "checkpoints" / "phase_a" / "eval_only_predictions.jsonl"
    )
    assert captured["model"] != str(tmp_path / "checkpoints" / "phase_a" / "best")


def test_eval_only_on_trained_phase_keeps_results_artifact(monkeypatch, tmp_path: Path):
    from chess_llm.training import train

    best_dir = tmp_path / "checkpoints" / "phase_a" / "best"
    best_dir.mkdir(parents=True)

    captured: dict[str, object] = {}

    def fake_run_eval(model_path, benchmark_dir, pred_path, **kwargs):
        captured["model"] = model_path
        captured["pred_path"] = pred_path
        return 0

    monkeypatch.setattr(train, "_run_eval", fake_run_eval)
    monkeypatch.setattr(train, "parse_args", lambda: _eval_only_args(tmp_path))

    assert train.main() == 0
    assert captured["model"] == str(best_dir)
    assert captured["pred_path"] == (
        tmp_path / "checkpoints" / "phase_a" / "eval_predictions.jsonl"
    )


def test_find_best_historical_baseline_takes_min_for_count_metrics(tmp_path: Path):
    import json

    from chess_llm.training.train import _find_best_historical_baseline

    phase_results = {
        "a": {"rules": {"legal_moves": 0.80, "missing_move_count": 40.0, "acpl": 250.0}},
        "b": {"rules": {"legal_moves": 0.70, "missing_move_count": 2.0, "acpl": 100.0}},
    }
    for phase_name, results in phase_results.items():
        phase_dir = tmp_path / f"phase_{phase_name}"
        phase_dir.mkdir(parents=True)
        (phase_dir / "eval_predictions.results.json").write_text(
            json.dumps(results), encoding="utf-8"
        )

    baseline_path = _find_best_historical_baseline(tmp_path, "c")

    assert baseline_path is not None
    merged = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert merged["rules"]["legal_moves"] == 0.80
    # Count metrics are lower-is-better: best-ever is the min, not the max.
    assert merged["rules"]["missing_move_count"] == 2.0
    assert merged["rules"]["acpl"] == 100.0


def test_train_cli_accepts_pure_bf16_flag(monkeypatch):
    from chess_llm.training import train

    monkeypatch.setattr(
        sys,
        "argv",
        ["chess-llm-train", "--phase", "a", "--pure-bf16"],
    )
    assert train.parse_args().pure_bf16 is True

    monkeypatch.setattr(sys, "argv", ["chess-llm-train", "--phase", "a"])
    assert train.parse_args().pure_bf16 is False


def test_training_weight_dtype_defaults_to_fp32_master_weights():
    from chess_llm.training.train import _training_torch_dtype

    assert _training_torch_dtype(False) == "float32"
    assert _training_torch_dtype(True) == "auto"


def test_resolve_resume_checkpoint_auto_uses_latest_numbered_checkpoint(tmp_path: Path):
    from chess_llm.training.train import _resolve_resume_checkpoint

    output_dir = tmp_path / "phase_a"
    (output_dir / "checkpoint-50").mkdir(parents=True)
    (output_dir / "checkpoint-100").mkdir()
    (output_dir / "checkpoint-final").mkdir()

    assert _resolve_resume_checkpoint(output_dir, "auto") == str(
        output_dir / "checkpoint-100",
    )


def test_resolve_resume_checkpoint_rejects_missing_path(tmp_path: Path):
    from chess_llm.training.train import _resolve_resume_checkpoint

    missing = tmp_path / "missing-checkpoint"

    try:
        _resolve_resume_checkpoint(tmp_path, str(missing))
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing resume checkpoint should fail")


def test_training_launcher_preflight_rejects_multi_gpu_without_torchrun(monkeypatch):
    from chess_llm.training import train

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
        ),
    )
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    error = train._training_launcher_preflight_error(torch_module=fake_torch)

    assert error is not None
    assert "torchrun --nproc_per_node=2" in error


def test_training_launcher_preflight_allows_torchrun_multi_gpu(monkeypatch):
    from chess_llm.training import train

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
        ),
    )
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")

    assert train._training_launcher_preflight_error(torch_module=fake_torch) is None


def test_resume_checkpoint_model_key_mismatch_is_a_hard_error(tmp_path: Path):
    from chess_llm.training import train

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.language_model.layers.0.weight": "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    model = types.SimpleNamespace(
        state_dict=lambda: {"model.layers.0.weight": object()},
    )

    error = train._resume_checkpoint_model_key_error(model, str(checkpoint))

    assert error is not None
    assert "missing model key" in error
    assert "unexpected checkpoint key" in error
    assert "model.language_model.layers.0.weight" in error


def test_resume_checkpoint_model_key_match_is_allowed(tmp_path: Path):
    from chess_llm.training import train

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.layers.0.weight": "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    model = types.SimpleNamespace(
        state_dict=lambda: {"model.layers.0.weight": object()},
    )

    assert train._resume_checkpoint_model_key_error(model, str(checkpoint)) is None


def test_package_train_eval_command_supports_full_benchmark_override(tmp_path: Path):
    from chess_llm.training.train import _build_eval_cmd

    default_cmd = _build_eval_cmd(
        "model-id",
        tmp_path / "benchmark",
        tmp_path / "predictions.jsonl",
        phase="a",
    )
    full_cmd = _build_eval_cmd(
        "model-id",
        tmp_path / "benchmark",
        tmp_path / "predictions.jsonl",
        phase="a",
        full_benchmark=True,
    )

    assert "--phase" in default_cmd
    assert "--full-benchmark" not in default_cmd
    assert "--full-benchmark" in full_cmd


def test_package_train_eval_command_forwards_wandb_offline_allowance(tmp_path: Path):
    from chess_llm.training.train import _build_eval_cmd

    cmd = _build_eval_cmd(
        "model-id",
        tmp_path / "benchmark",
        tmp_path / "predictions.jsonl",
        no_wandb=False,
        allow_wandb_offline=True,
    )

    assert "--wandb-project" in cmd
    assert "--allow-wandb-offline" in cmd


def test_package_curriculum_train_command_uses_module_entrypoint(tmp_path: Path):
    from chess_llm.training.run_curriculum import _build_train_phase_cmd

    args = Namespace(
        data_root=tmp_path / "data",
        output_root=tmp_path / "checkpoints",
        benchmark_dir=tmp_path / "benchmark",
        inference_backend="transformers",
        attn_implementation="auto",
        eval_batch_size=16,
        eval_max_new_tokens=256,
        eval_acpl_depth=20,
        stockfish_path=None,
        smoke_run=False,
        max_train_examples=None,
        task_upsample=[],
        max_eval_examples=None,
        max_benchmark_examples_per_split=None,
        max_steps=None,
        trainer_eval_steps=None,
        trainer_save_steps=None,
        skip_trainer_eval=False,
        require_phase_gate=False,
        no_acpl=False,
        full_acpl_report=False,
        no_wandb=True,
        allow_wandb_offline=False,
        wandb_project="chess-sft",
    )

    cmd = _build_train_phase_cmd(
        "a",
        args,
        curriculum_id="curriculum-test",
        wandb_group=None,
    )

    assert cmd[:4] == [
        sys.executable,
        "-m",
        "chess_llm.training.train",
        "--phase",
    ]


def test_package_curriculum_train_command_forwards_task_upsample(tmp_path: Path):
    from chess_llm.training.run_curriculum import _build_train_phase_cmd

    args = Namespace(
        data_root=tmp_path / "data",
        output_root=tmp_path / "checkpoints",
        benchmark_dir=tmp_path / "benchmark",
        inference_backend="transformers",
        attn_implementation="auto",
        eval_batch_size=16,
        eval_max_new_tokens=192,
        eval_acpl_depth=20,
        stockfish_path=None,
        smoke_run=False,
        max_train_examples=None,
        task_upsample=["1.5_state_tracking=4", "1.9_fen_assembly=4"],
        max_eval_examples=None,
        max_benchmark_examples_per_split=100,
        max_steps=None,
        trainer_eval_steps=None,
        trainer_save_steps=None,
        skip_trainer_eval=True,
        require_phase_gate=False,
        no_acpl=True,
        full_acpl_report=False,
        no_wandb=True,
        allow_wandb_offline=False,
        wandb_project="chess-sft",
    )

    cmd = _build_train_phase_cmd(
        "a",
        args,
        curriculum_id="curriculum-test",
        wandb_group=None,
    )

    assert cmd.count("--task-upsample") == 2
    assert ["--task-upsample", "1.5_state_tracking=4"] == cmd[
        cmd.index("--task-upsample"): cmd.index("--task-upsample") + 2
    ]
    assert "1.9_fen_assembly=4" in cmd
    assert "--skip-trainer-eval" in cmd
    assert "--no-acpl" in cmd


def test_package_curriculum_train_command_forwards_data_transform_flags(tmp_path: Path):
    from chess_llm.training.run_curriculum import _build_train_phase_cmd

    args = Namespace(
        data_root=tmp_path / "data",
        output_root=tmp_path / "checkpoints",
        benchmark_dir=tmp_path / "benchmark",
        inference_backend="transformers",
        attn_implementation="auto",
        eval_batch_size=16,
        eval_max_new_tokens=192,
        eval_acpl_depth=20,
        stockfish_path=None,
        smoke_run=False,
        max_train_examples=None,
        task_upsample=[],
        task_include=["7.1_best_move_selection", "7.2_puzzle_solving"],
        task_exclude=["7.8_candidate_ratings"],
        move_only_task=["7.1_best_move_selection"],
        max_eval_examples=None,
        max_benchmark_examples_per_split=100,
        max_steps=None,
        learning_rate=None,
        trainer_eval_steps=None,
        trainer_save_steps=None,
        skip_trainer_eval=True,
        gradient_checkpointing=False,
        enable_liger_fused_linear_ce=False,
        require_phase_gate=False,
        no_acpl=True,
        full_acpl_report=False,
        no_wandb=True,
        allow_wandb_offline=False,
        wandb_project="chess-sft",
    )

    cmd = _build_train_phase_cmd(
        "c",
        args,
        curriculum_id="curriculum-test",
        wandb_group=None,
    )

    assert cmd.count("--task-include") == 2
    assert "7.2_puzzle_solving" in cmd
    assert ["--task-exclude", "7.8_candidate_ratings"] == cmd[
        cmd.index("--task-exclude"): cmd.index("--task-exclude") + 2
    ]
    assert ["--move-only-task", "7.1_best_move_selection"] == cmd[
        cmd.index("--move-only-task"): cmd.index("--move-only-task") + 2
    ]


def test_package_curriculum_train_command_forwards_wandb_offline_allowance(tmp_path: Path):
    from chess_llm.training.run_curriculum import _build_train_phase_cmd

    args = Namespace(
        data_root=tmp_path / "data",
        output_root=tmp_path / "checkpoints",
        benchmark_dir=tmp_path / "benchmark",
        inference_backend="transformers",
        attn_implementation="auto",
        eval_batch_size=16,
        eval_max_new_tokens=192,
        eval_acpl_depth=20,
        stockfish_path=None,
        smoke_run=False,
        max_train_examples=None,
        task_upsample=[],
        max_eval_examples=None,
        max_benchmark_examples_per_split=None,
        max_steps=None,
        trainer_eval_steps=None,
        trainer_save_steps=None,
        skip_trainer_eval=False,
        require_phase_gate=False,
        no_acpl=True,
        full_acpl_report=False,
        no_wandb=False,
        allow_wandb_offline=True,
        wandb_project="chess-sft",
    )

    cmd = _build_train_phase_cmd(
        "a",
        args,
        curriculum_id="curriculum-test",
        wandb_group="curriculum-test",
    )

    assert "--wandb-project" in cmd
    assert "--allow-wandb-offline" in cmd


def test_package_curriculum_pre_eval_command_is_phase_aware_and_bounded(tmp_path: Path):
    from chess_llm.training.run_curriculum import (
        DEFAULT_PRE_EVAL_MAX_EXAMPLES_PER_SPLIT,
        _build_pre_eval_cmd,
    )

    args = Namespace(
        output_root=tmp_path / "checkpoints",
        benchmark_dir=tmp_path / "benchmark",
        inference_backend="transformers",
        attn_implementation="auto",
        eval_batch_size=16,
        eval_max_new_tokens=192,
        eval_acpl_depth=20,
        stockfish_path=None,
        max_benchmark_examples_per_split=None,
        wandb_project="chess-sft",
        no_wandb=True,
        allow_wandb_offline=False,
    )

    cmd = _build_pre_eval_cmd(
        "a",
        "model-id",
        args,
        curriculum_id="curriculum-test",
        wandb_group=None,
    )

    assert ["--phase", "a"] == cmd[cmd.index("--phase"): cmd.index("--phase") + 2]
    assert "--report-only" in cmd
    assert "--no-acpl" in cmd
    assert ["--max-examples-per-split", str(DEFAULT_PRE_EVAL_MAX_EXAMPLES_PER_SPLIT)] == cmd[
        cmd.index("--max-examples-per-split"): cmd.index("--max-examples-per-split") + 2
    ]


def test_package_curriculum_pre_eval_respects_explicit_benchmark_cap(tmp_path: Path):
    from chess_llm.training.run_curriculum import _build_pre_eval_cmd

    args = Namespace(
        output_root=tmp_path / "checkpoints",
        benchmark_dir=tmp_path / "benchmark",
        inference_backend="transformers",
        attn_implementation="auto",
        eval_batch_size=16,
        eval_max_new_tokens=192,
        eval_acpl_depth=20,
        stockfish_path=None,
        max_benchmark_examples_per_split=7,
        wandb_project="chess-sft",
        no_wandb=True,
        allow_wandb_offline=False,
    )

    cmd = _build_pre_eval_cmd(
        "b",
        "model-id",
        args,
        curriculum_id="curriculum-test",
        wandb_group=None,
    )

    assert ["--phase", "b"] == cmd[cmd.index("--phase"): cmd.index("--phase") + 2]
    assert ["--max-examples-per-split", "7"] == cmd[
        cmd.index("--max-examples-per-split"): cmd.index("--max-examples-per-split") + 2
    ]


def test_training_cli_logging_suppresses_noisy_dependency_info():
    from chess_llm.training.logging_utils import (
        NOISY_DEPENDENCY_LOGGERS,
        configure_cli_logging,
    )

    for logger_name in NOISY_DEPENDENCY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.NOTSET)

    configure_cli_logging()

    for logger_name in NOISY_DEPENDENCY_LOGGERS:
        assert logging.getLogger(logger_name).level == logging.WARNING


def test_training_cli_warning_filters_expected_liger_token_accuracy_warning():
    from chess_llm.training.logging_utils import configure_cli_logging

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        configure_cli_logging()
        warnings.warn(
            "liger-kernel did not return token_accuracy when requested. "
            "The mean_token_accuracy metric will not be logged.",
            UserWarning,
        )
        warnings.warn("unrelated warning", UserWarning)

    assert [str(warning.message) for warning in captured] == ["unrelated warning"]


def test_qwen35_uses_qwen_safe_liger_config(monkeypatch, tmp_path: Path):
    class FakeSFTConfig:
        def __init__(
            self,
            output_dir=None,
            run_name=None,
            packing=None,
            assistant_only_loss=None,
            num_train_epochs=None,
            learning_rate=None,
            warmup_ratio=None,
            weight_decay=None,
            lr_scheduler_type=None,
            per_device_train_batch_size=None,
            per_device_eval_batch_size=None,
            gradient_accumulation_steps=None,
            bf16=None,
            gradient_checkpointing=None,
            gradient_checkpointing_kwargs=None,
            use_liger_kernel=None,
            liger_kernel_config=None,
            tf32=None,
            dataset_num_proc=None,
            dataloader_persistent_workers=None,
            dataloader_prefetch_factor=None,
            pad_to_multiple_of=None,
            padding_free=None,
            eval_strategy=None,
            eval_steps=None,
            save_strategy=None,
            save_steps=None,
            save_total_limit=None,
            load_best_model_at_end=None,
            metric_for_best_model=None,
            greater_is_better=None,
            save_safetensors=None,
            save_only_model=None,
            logging_steps=None,
            report_to=None,
            dataloader_num_workers=None,
            seed=None,
            dataloader_pin_memory=None,
            include_num_input_tokens_seen=None,
            max_length=None,
            eos_token=None,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig))
    _expire_training_args(monkeypatch)

    from chess_llm.training.phases import PHASE_A
    from chess_llm.training.training_args import build_sft_config

    cfg = build_sft_config(PHASE_A, tmp_path, model_type="qwen3_5")

    assert cfg.kwargs["use_liger_kernel"] is True
    assert cfg.kwargs["liger_kernel_config"] == {
        "cross_entropy": False,
        "fused_linear_cross_entropy": False,
    }
    assert cfg.kwargs["tf32"] is True
    assert cfg.kwargs["dataset_num_proc"] >= 1
    assert cfg.kwargs["dataloader_persistent_workers"] is True
    assert cfg.kwargs["dataloader_prefetch_factor"] == 2
    assert cfg.kwargs["pad_to_multiple_of"] == 8
    assert cfg.kwargs["include_num_input_tokens_seen"] is True
    assert cfg.kwargs["packing"] is False
    assert cfg.kwargs["padding_free"] is False


def test_build_sft_config_accepts_learning_rate_override(monkeypatch, tmp_path: Path):
    class FakeSFTConfig:
        def __init__(
            self,
            output_dir=None,
            run_name=None,
            assistant_only_loss=None,
            num_train_epochs=None,
            learning_rate=None,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig))
    _expire_training_args(monkeypatch)

    from chess_llm.training.phases import PHASE_B
    from chess_llm.training.training_args import build_sft_config

    cfg = build_sft_config(PHASE_B, tmp_path, learning_rate=2e-6)

    assert cfg.kwargs["learning_rate"] == 2e-6


def test_build_sft_config_uses_flash_attention_packing_when_available(
    monkeypatch,
    tmp_path: Path,
):
    class FakeSFTConfig:
        def __init__(
            self,
            output_dir=None,
            run_name=None,
            packing=None,
            packing_strategy=None,
            padding_free=None,
            pad_to_multiple_of=None,
            assistant_only_loss=None,
            tf32=None,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig))
    _expire_training_args(monkeypatch)

    from chess_llm.training.phases import PHASE_A
    from chess_llm.training.training_args import build_sft_config

    for attn_implementation in [
        "flash_attention_2",
        "flash_attention_4",
        "kernels-community/flash-attn2@bcc70a66fbe0445d4484f56167d706134d53633d",
    ]:
        cfg = build_sft_config(
            PHASE_A,
            tmp_path,
            attn_implementation=attn_implementation,
        )

        assert cfg.kwargs["packing"] is True
        assert cfg.kwargs["packing_strategy"] == "bfd"
        assert cfg.kwargs["padding_free"] is True
        assert cfg.kwargs["pad_to_multiple_of"] == 8
        assert cfg.kwargs["tf32"] is True


def test_build_sft_config_can_force_packing_without_padding_free(
    monkeypatch,
    tmp_path: Path,
):
    class FakeSFTConfig:
        def __init__(
            self,
            output_dir=None,
            run_name=None,
            packing=None,
            packing_strategy=None,
            padding_free=None,
            max_length=None,
            pad_to_multiple_of=None,
            assistant_only_loss=None,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig))
    _expire_training_args(monkeypatch)

    from chess_llm.training.phases import PHASE_A
    from chess_llm.training.training_args import build_sft_config

    cfg = build_sft_config(
        PHASE_A,
        tmp_path,
        attn_implementation="sdpa",
        packing="on",
        max_length=1024,
    )

    assert cfg.kwargs["packing"] is True
    assert cfg.kwargs["packing_strategy"] == "bfd"
    assert cfg.kwargs["padding_free"] is False
    assert cfg.kwargs["max_length"] == 1024


def test_build_sft_config_can_disable_flash_attention_packing(
    monkeypatch,
    tmp_path: Path,
):
    class FakeSFTConfig:
        def __init__(
            self,
            output_dir=None,
            run_name=None,
            packing=None,
            packing_strategy=None,
            padding_free=None,
            assistant_only_loss=None,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig))
    _expire_training_args(monkeypatch)

    from chess_llm.training.phases import PHASE_A
    from chess_llm.training.training_args import build_sft_config

    cfg = build_sft_config(
        PHASE_A,
        tmp_path,
        attn_implementation="flash_attention_2",
        packing="off",
    )

    assert cfg.kwargs["packing"] is False
    assert cfg.kwargs["padding_free"] is False
    assert "packing_strategy" not in cfg.kwargs or cfg.kwargs["packing_strategy"] is None


def test_qwen35_can_opt_into_liger_fused_linear_ce(monkeypatch, tmp_path: Path):
    class FakeSFTConfig:
        def __init__(
            self,
            output_dir=None,
            run_name=None,
            assistant_only_loss=None,
            use_liger_kernel=None,
            liger_kernel_config=None,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig))
    _expire_training_args(monkeypatch)

    from chess_llm.training.phases import PHASE_A
    from chess_llm.training.training_args import build_sft_config

    cfg = build_sft_config(
        PHASE_A,
        tmp_path,
        model_type="qwen3_5",
        liger_fused_linear_cross_entropy=True,
    )

    assert cfg.kwargs["liger_kernel_config"] == {
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
    }


def test_training_step_estimate_accounts_for_accumulation_and_world_size(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "trl",
        sys.modules.get("trl")
        or types.SimpleNamespace(SFTConfig=type("FakeSFTConfig", (), {})),
    )
    _expire_training_args(monkeypatch)

    from chess_llm.training.phases import PHASE_A
    from chess_llm.training.training_args import estimate_training_steps

    assert estimate_training_steps(
        PHASE_A,
        train_dataset_size=3200,
        world_size=2,
    ) == 150
    assert estimate_training_steps(
        PHASE_A,
        train_dataset_size=3200,
        max_steps=10,
        world_size=2,
    ) == 10
    assert estimate_training_steps(
        PHASE_A,
        train_dataset_size=3200,
        num_train_epochs=1,
        world_size=2,
    ) == 50


def test_build_sft_config_honors_num_train_epochs_override(
    monkeypatch,
    tmp_path: Path,
):
    class FakeSFTConfig:
        def __init__(
            self,
            output_dir=None,
            run_name=None,
            packing=None,
            assistant_only_loss=None,
            num_train_epochs=None,
            warmup_steps=None,
            warmup_ratio=None,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig))
    _expire_training_args(monkeypatch)

    from chess_llm.training.phases import PHASE_A
    from chess_llm.training.training_args import build_sft_config

    cfg = build_sft_config(
        PHASE_A,
        tmp_path,
        num_train_epochs=1,
    )

    assert cfg.kwargs["num_train_epochs"] == 1
    assert cfg.kwargs["warmup_ratio"] == PHASE_A.warmup_ratio
    assert cfg.kwargs["warmup_steps"] is None


def test_build_sft_config_uses_warmup_ratio_not_example_count_estimates(
    monkeypatch,
    tmp_path: Path,
):
    class FakeSFTConfig:
        def __init__(
            self,
            output_dir=None,
            run_name=None,
            packing=None,
            assistant_only_loss=None,
            num_train_epochs=None,
            warmup_steps=None,
            warmup_ratio=None,
            max_steps=None,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig))
    _expire_training_args(monkeypatch)

    from chess_llm.training.phases import PHASE_A
    from chess_llm.training.training_args import build_sft_config

    cfg = build_sft_config(
        PHASE_A,
        tmp_path,
        max_steps=10,
    )

    # Warmup must track the Trainer's actual optimizer steps (packing-aware),
    # so build_sft_config passes warmup_ratio and never a derived step count.
    assert cfg.kwargs["warmup_ratio"] == 0.03
    assert cfg.kwargs["warmup_steps"] is None
    assert cfg.kwargs["max_steps"] == 10


def test_skip_trainer_eval_disables_step_checkpoint_saves_by_default(monkeypatch, tmp_path: Path):
    class FakeSFTConfig:
        def __init__(
            self,
            output_dir=None,
            run_name=None,
            packing=None,
            assistant_only_loss=None,
            eval_strategy=None,
            eval_steps=None,
            save_strategy=None,
            save_steps=None,
            save_total_limit=None,
            load_best_model_at_end=None,
            metric_for_best_model=None,
            greater_is_better=None,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig))
    _expire_training_args(monkeypatch)

    from chess_llm.training.phases import PHASE_A
    from chess_llm.training.training_args import build_sft_config

    cfg = build_sft_config(PHASE_A, tmp_path, trainer_eval=False)

    assert cfg.kwargs["eval_strategy"] == "no"
    assert cfg.kwargs["eval_steps"] is None
    assert cfg.kwargs["save_strategy"] == "no"
    assert cfg.kwargs["save_steps"] is None
    assert cfg.kwargs["save_total_limit"] is None
    assert cfg.kwargs["load_best_model_at_end"] is False
    assert cfg.kwargs["metric_for_best_model"] is None
    assert cfg.kwargs["greater_is_better"] is None


def test_skip_trainer_eval_can_still_save_periodic_checkpoints(monkeypatch, tmp_path: Path):
    class FakeSFTConfig:
        def __init__(
            self,
            output_dir=None,
            run_name=None,
            packing=None,
            assistant_only_loss=None,
            eval_strategy=None,
            eval_steps=None,
            save_strategy=None,
            save_steps=None,
            save_total_limit=None,
            load_best_model_at_end=None,
            metric_for_best_model=None,
            greater_is_better=None,
            save_only_model=None,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig))
    _expire_training_args(monkeypatch)

    from chess_llm.training.phases import PHASE_A
    from chess_llm.training.training_args import build_sft_config

    cfg = build_sft_config(
        PHASE_A,
        tmp_path,
        trainer_eval=False,
        save_steps=250,
    )

    assert cfg.kwargs["eval_strategy"] == "no"
    assert cfg.kwargs["eval_steps"] is None
    assert cfg.kwargs["save_strategy"] == "steps"
    assert cfg.kwargs["save_steps"] == 250
    assert cfg.kwargs["save_total_limit"] == 3
    assert cfg.kwargs["save_only_model"] is False
    assert cfg.kwargs["load_best_model_at_end"] is False
    assert cfg.kwargs["metric_for_best_model"] is None
    assert cfg.kwargs["greater_is_better"] is None


def test_bounded_skip_trainer_eval_does_not_auto_add_midrun_saves():
    from chess_llm.training import train

    args = Namespace(
        smoke_run=False,
        max_train_examples=None,
        max_eval_examples=None,
        max_benchmark_examples_per_split=None,
        max_steps=50,
        num_train_epochs=1,
        trainer_eval_steps=None,
        trainer_save_steps=None,
        skip_trainer_eval=True,
    )

    overrides = train._resolve_run_overrides(args)

    assert overrides.max_steps == 50
    assert overrides.num_train_epochs == 1
    assert overrides.eval_steps is None
    assert overrides.save_steps is None
    assert overrides.logging_steps == 5


def test_bounded_trainer_eval_keeps_short_eval_save_cadence():
    from chess_llm.training import train

    args = Namespace(
        smoke_run=False,
        max_train_examples=None,
        max_eval_examples=None,
        max_benchmark_examples_per_split=None,
        max_steps=50,
        trainer_eval_steps=None,
        trainer_save_steps=None,
        skip_trainer_eval=False,
    )

    overrides = train._resolve_run_overrides(args)

    assert overrides.eval_steps == 25
    assert overrides.save_steps == 25
    assert overrides.logging_steps == 5


def test_bounded_short_run_cadence_applies_up_to_threshold():
    from chess_llm.training import train

    args = Namespace(
        smoke_run=False,
        max_train_examples=None,
        max_eval_examples=None,
        max_benchmark_examples_per_split=None,
        max_steps=train.SHORT_RUN_CADENCE_MAX_STEPS,
        trainer_eval_steps=None,
        trainer_save_steps=None,
        skip_trainer_eval=False,
    )

    overrides = train._resolve_run_overrides(args)

    assert overrides.eval_steps == 50
    assert overrides.save_steps == 50
    assert overrides.logging_steps == 25


def _schedule_args(tmp_path: Path, **overrides) -> Namespace:
    ns = Namespace(
        phase="schedule",
        data_root=tmp_path / "data",
        output_root=tmp_path / "checkpoints",
        benchmark_dir=tmp_path / "benchmark",
        base_model=None,
        dry_run=False,
        smoke_run=False,
        eval_only=False,
        skip_eval=True,
        require_phase_gate=False,
        max_train_examples=None,
        task_upsample=[],
        max_eval_examples=None,
        max_benchmark_examples_per_split=None,
        full_benchmark_eval=False,
        max_steps=None,
        num_train_epochs=None,
        trainer_eval_steps=None,
        trainer_save_steps=None,
        skip_trainer_eval=False,
        resume_from_checkpoint=None,
        wandb_project="chess-sft",
        no_wandb=True,
        allow_wandb_offline=False,
        run_name=None,
        wandb_group=None,
        inference_backend="transformers",
        attn_implementation="auto",
        packing="auto",
        max_length=2048,
        schedule_total_examples=None,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def test_train_cli_accepts_schedule_phase_and_budget(monkeypatch):
    from chess_llm.training import train

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chess-llm-train",
            "--phase",
            "schedule",
            "--schedule-total-examples",
            "500000",
        ],
    )

    args = train.parse_args()

    assert args.phase == "schedule"
    assert args.schedule_total_examples == 500000


def test_schedule_mode_rejects_incompatible_flags(monkeypatch, tmp_path: Path):
    from chess_llm.training import train

    for overrides in (
        {"require_phase_gate": True},
        {"packing": "on"},
        {"num_train_epochs": 3.0},
    ):
        monkeypatch.setattr(
            train,
            "parse_args",
            lambda overrides=overrides: _schedule_args(tmp_path, **overrides),
        )
        assert train.main() == 2, overrides


def test_schedule_mode_config_error_allows_compatible_flags():
    from chess_llm.training import train

    assert train._schedule_mode_config_error(
        Namespace(require_phase_gate=False, packing="auto", num_train_epochs=None)
    ) is None
    assert train._schedule_mode_config_error(
        Namespace(require_phase_gate=False, packing="off", num_train_epochs=1.0)
    ) is None


def test_resolve_schedule_total_examples_takes_min_of_caps():
    from chess_llm.training.train import _resolve_schedule_total_examples

    assert _resolve_schedule_total_examples(None, None) is None
    assert _resolve_schedule_total_examples(100_000, None) == 100_000
    assert _resolve_schedule_total_examples(None, 50_000) == 50_000
    assert _resolve_schedule_total_examples(100_000, 50_000) == 50_000
    assert _resolve_schedule_total_examples(20_000, 50_000) == 20_000


def test_schedule_training_budget_flows_into_builder_not_limit_dataset(
    monkeypatch,
    tmp_path: Path,
):
    import json

    from chess_llm.training import train
    from chess_llm.training.data import mixer
    from chess_llm.training.schedule import SCHEDULE_V1, SegmentPlan

    plans = [
        SegmentPlan(
            index=0,
            name="only",
            start_row=0,
            end_row=64,
            tier_weights=((1, 1.0),),
            tier_rows=((1, 64),),
        ),
    ]

    class FakeDataset:
        def __init__(self, size: int):
            self.size = size

        def __len__(self):
            return self.size

    captured: dict[str, object] = {}

    def fake_build_schedule_dataset(
        schedule,
        data_root,
        eval_fraction=0.02,
        seed=42,
        task_upsample=None,
        total_examples=None,
    ):
        captured["total_examples"] = total_examples
        return FakeDataset(64), FakeDataset(4), plans

    monkeypatch.setattr(mixer, "build_schedule_dataset", fake_build_schedule_dataset)

    limit_calls: list[object] = []

    def fake_limit(ds, max_examples, *, seed=42):
        limit_calls.append(max_examples)
        return ds

    monkeypatch.setattr(train, "_limit_dataset", fake_limit)

    output_dir = tmp_path / "phase_schedule"
    output_dir.mkdir(parents=True)
    args = _schedule_args(tmp_path, schedule_total_examples=100)
    overrides = train.RunOverrides(max_train_examples=64)

    train_ds, eval_ds, out_plans, boundaries = train._build_schedule_training_data(
        SCHEDULE_V1,
        args,
        overrides,
        {},
        output_dir,
    )

    # min(--schedule-total-examples, --max-train-examples) becomes the budget;
    # the ordered train split must never pass through _limit_dataset.
    assert captured["total_examples"] == 64
    assert limit_calls == []
    assert out_plans == plans
    assert boundaries == [2]

    payload = json.loads(
        (output_dir / "schedule_plan.json").read_text(encoding="utf-8")
    )
    assert payload["schedule"] == "schedule"
    assert payload["seed"] == 42
    assert payload["total_examples"] == 64
    assert payload["boundary_steps"] == [2]
    assert payload["segments"][0]["tier_rows"] == {"1": 64}


def test_schedule_dry_run_caps_budget_by_max_train_examples(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.training import train
    from chess_llm.training.data import mixer

    captured: dict[str, object] = {}

    def fake_summarize(schedule, data_root, task_upsample=None, total_examples=None):
        captured["total_examples"] = total_examples
        return {
            "total_examples": total_examples,
            "tier_pool_sizes": {"tier_1": 1000},
            "boundary_steps": [2],
            "segments": [
                {
                    "index": 0,
                    "name": "only",
                    "start_row": 0,
                    "end_row": total_examples,
                    "rows": total_examples,
                    "tier_rows": {"tier_1": total_examples},
                },
            ],
        }

    monkeypatch.setattr(mixer, "summarize_schedule_data", fake_summarize)
    monkeypatch.setattr(
        train,
        "parse_args",
        lambda: _schedule_args(
            tmp_path,
            dry_run=True,
            schedule_total_examples=100,
            max_train_examples=50,
        ),
    )

    assert train.main() == 0
    assert captured["total_examples"] == 50


def test_build_sft_config_sequential_dataset_sets_sampler_fields(
    monkeypatch,
    tmp_path: Path,
):
    class FakeSFTConfig:
        def __init__(
            self,
            output_dir=None,
            run_name=None,
            packing=None,
            assistant_only_loss=None,
            train_sampling_strategy=None,
            shuffle_dataset=None,
            padding_free=None,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig))
    _expire_training_args(monkeypatch)

    from chess_llm.training.phases import PHASE_A
    from chess_llm.training.training_args import build_sft_config

    cfg = build_sft_config(PHASE_A, tmp_path, packing="off", sequential_dataset=True)

    assert cfg.kwargs["train_sampling_strategy"] == "sequential"
    assert cfg.kwargs["shuffle_dataset"] is False
    assert cfg.kwargs["packing"] is False


def test_build_sft_config_sequential_requires_sampling_strategy_field(
    monkeypatch,
    tmp_path: Path,
):
    class FakeSFTConfig:
        """Stub without train_sampling_strategy/shuffle_dataset fields."""

        def __init__(
            self,
            output_dir=None,
            run_name=None,
            packing=None,
            assistant_only_loss=None,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig))
    _expire_training_args(monkeypatch)

    from chess_llm.training.phases import PHASE_A
    from chess_llm.training.training_args import build_sft_config

    # _filter_supported_sft_kwargs would silently drop the kwarg and degrade
    # to a RandomSampler, destroying the schedule — this must fail loudly.
    try:
        build_sft_config(PHASE_A, tmp_path, packing="off", sequential_dataset=True)
    except RuntimeError as exc:
        assert "train_sampling_strategy" in str(exc)
    else:
        raise AssertionError("sequential_dataset with old TRL should raise RuntimeError")


def test_build_sft_config_sequential_rejects_packing(monkeypatch, tmp_path: Path):
    class FakeSFTConfig:
        def __init__(
            self,
            output_dir=None,
            run_name=None,
            packing=None,
            packing_strategy=None,
            assistant_only_loss=None,
            train_sampling_strategy=None,
            shuffle_dataset=None,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig))
    _expire_training_args(monkeypatch)

    from chess_llm.training.phases import PHASE_A
    from chess_llm.training.training_args import build_sft_config

    try:
        build_sft_config(
            PHASE_A,
            tmp_path,
            attn_implementation="sdpa",
            packing="on",
            sequential_dataset=True,
        )
    except ValueError as exc:
        assert "packing" in str(exc)
    else:
        raise AssertionError("sequential_dataset with packing on should raise ValueError")


def _load_schedule_callback(monkeypatch):
    fake_transformers = sys.modules.get("transformers") or types.SimpleNamespace(
        TrainerCallback=type("TrainerCallback", (), {}),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    _expire_module(monkeypatch, "chess_llm.training.schedule_callback")
    return importlib.import_module("chess_llm.training.schedule_callback")


def _callback_plans():
    from chess_llm.training.schedule import SegmentPlan

    return [
        SegmentPlan(
            index=0,
            name="mechanics",
            start_row=0,
            end_row=64,
            tier_weights=((1, 0.5), (2, 0.5)),
            tier_rows=((1, 32), (2, 32)),
        ),
        SegmentPlan(
            index=1,
            name="planning",
            start_row=64,
            end_row=128,
            tier_weights=((1, 0.2), (7, 0.8)),
            tier_rows=((1, 13), (7, 51)),
        ),
    ]


def test_mixing_schedule_callback_injects_schedule_logs(monkeypatch):
    module = _load_schedule_callback(monkeypatch)

    callback = module.MixingScheduleCallback(_callback_plans(), [2, 4])

    logs: dict[str, object] = {}
    callback.on_log(
        None,
        types.SimpleNamespace(global_step=1),
        types.SimpleNamespace(),
        logs=logs,
    )
    assert logs["schedule/segment_index"] == 0
    assert logs["schedule/tier_weight_1"] == 0.5
    assert logs["schedule/tier_weight_2"] == 0.5
    assert logs["schedule/tier_weight_7"] == 0.0
    assert logs["schedule/replay_fraction_t12"] == 1.0
    assert logs["schedule/segment_progress"] == 0.5

    late_logs: dict[str, object] = {}
    callback.on_log(
        None,
        types.SimpleNamespace(global_step=3),
        types.SimpleNamespace(),
        logs=late_logs,
    )
    assert late_logs["schedule/segment_index"] == 1
    assert late_logs["schedule/tier_weight_7"] == 0.8
    assert late_logs["schedule/replay_fraction_t12"] == 0.2
    assert late_logs["schedule/segment_progress"] == 0.5


def test_mixing_schedule_callback_forces_saves_at_interior_boundaries(monkeypatch):
    module = _load_schedule_callback(monkeypatch)

    callback = module.MixingScheduleCallback(_callback_plans(), [2, 4])

    control = types.SimpleNamespace(should_save=False, should_evaluate=False)
    callback.on_step_end(None, types.SimpleNamespace(global_step=1), control)
    assert control.should_save is False
    assert control.should_evaluate is False

    callback.on_step_end(None, types.SimpleNamespace(global_step=2), control)
    assert control.should_save is True
    assert control.should_evaluate is True

    # The final boundary coincides with the end of training, where the
    # Trainer saves/evaluates through its own end-of-train path.
    final_control = types.SimpleNamespace(should_save=False, should_evaluate=False)
    callback.on_step_end(None, types.SimpleNamespace(global_step=4), final_control)
    assert final_control.should_save is False
    assert final_control.should_evaluate is False

    # With trainer eval disabled, boundaries must not force an evaluation
    # (the Trainer would crash evaluating without an eval dataset).
    no_eval_callback = module.MixingScheduleCallback(
        _callback_plans(), [2, 4], evaluate_at_boundaries=False,
    )
    no_eval_control = types.SimpleNamespace(should_save=False, should_evaluate=False)
    no_eval_callback.on_step_end(None, types.SimpleNamespace(global_step=2), no_eval_control)
    assert no_eval_control.should_save is True
    assert no_eval_control.should_evaluate is False


def test_mixing_schedule_callback_warns_on_max_steps_drift(monkeypatch, caplog):
    module = _load_schedule_callback(monkeypatch)

    callback = module.MixingScheduleCallback(_callback_plans(), [2, 4])

    with caplog.at_level(logging.WARNING, logger=module.logger.name):
        callback.on_train_begin(
            None,
            types.SimpleNamespace(max_steps=10),
            types.SimpleNamespace(),
        )
    assert any("Schedule drift" in record.getMessage() for record in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=module.logger.name):
        callback.on_train_begin(
            None,
            types.SimpleNamespace(max_steps=4),
            types.SimpleNamespace(),
        )
    assert not caplog.records


def test_promote_callback_to_front_moves_before_integrations():
    from chess_llm.training import train

    sentinel = object()
    handler = types.SimpleNamespace(callbacks=["progress", "wandb", sentinel])
    trainer = types.SimpleNamespace(callback_handler=handler)

    assert train._promote_callback_to_front(trainer, sentinel) is True
    assert handler.callbacks[0] is sentinel

    missing = types.SimpleNamespace(callback_handler=None)
    assert train._promote_callback_to_front(missing, sentinel) is False


def test_find_best_historical_baseline_merges_all_phases_for_schedule(tmp_path: Path):
    import json

    from chess_llm.training.train import _find_best_historical_baseline

    phase_results = {
        "a": {"rules": {"legal_moves": 0.80}},
        "c": {"rules": {"legal_moves": 0.90}},
    }
    for phase_name, results in phase_results.items():
        phase_dir = tmp_path / f"phase_{phase_name}"
        phase_dir.mkdir(parents=True)
        (phase_dir / "eval_predictions.results.json").write_text(
            json.dumps(results), encoding="utf-8"
        )

    baseline_path = _find_best_historical_baseline(tmp_path, "schedule")

    assert baseline_path is not None
    merged = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert merged["rules"]["legal_moves"] == 0.90


def test_long_bounded_runs_keep_default_eval_save_cadence():
    from chess_llm.training import train

    args = Namespace(
        smoke_run=False,
        max_train_examples=None,
        max_eval_examples=None,
        max_benchmark_examples_per_split=None,
        max_steps=5000,
        trainer_eval_steps=None,
        trainer_save_steps=None,
        skip_trainer_eval=False,
    )

    overrides = train._resolve_run_overrides(args)

    # Long bounded runs must not be forced onto the <=50-step smoke cadence;
    # leaving these None defers to the normal configured trainer defaults.
    assert overrides.max_steps == 5000
    assert overrides.eval_steps is None
    assert overrides.save_steps is None
    assert overrides.logging_steps is None
