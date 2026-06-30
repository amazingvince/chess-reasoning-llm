from __future__ import annotations

import builtins
import importlib
import logging
from argparse import Namespace
from pathlib import Path
import sys
import types
import warnings


def test_training_entrypoint_modules_import_without_heavy_gpu_deps(monkeypatch):
    for module_name in list(sys.modules):
        if module_name in {
            "chess_llm.training.evaluate",
            "chess_llm.training.run_curriculum",
            "chess_llm.training.train",
            "torch",
            "transformers",
            "tqdm",
            "trl",
            "vllm",
        }:
            sys.modules.pop(module_name, None)

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


def test_legacy_training_wrappers_alias_package_modules(monkeypatch):
    legacy_dir = Path(__file__).resolve().parents[1] / "sft" / "training"
    monkeypatch.syspath_prepend(str(legacy_dir))

    for module_name in ["evaluate", "run_curriculum", "train"]:
        sys.modules.pop(module_name, None)

    legacy_train = importlib.import_module("train")
    legacy_evaluate = importlib.import_module("evaluate")
    legacy_curriculum = importlib.import_module("run_curriculum")

    assert legacy_train is importlib.import_module("chess_llm.training.train")
    assert legacy_evaluate is importlib.import_module("chess_llm.training.evaluate")
    assert legacy_curriculum is importlib.import_module("chess_llm.training.run_curriculum")


def test_legacy_training_args_wrapper_aliases_package_module(monkeypatch):
    legacy_dir = Path(__file__).resolve().parents[1] / "sft" / "training"
    monkeypatch.syspath_prepend(str(legacy_dir))

    class FakeSFTConfig:
        pass

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig))
    for module_name in [
        "config",
        "config.training_args",
        "chess_llm.training.training_args",
    ]:
        sys.modules.pop(module_name, None)

    legacy = importlib.import_module("config.training_args")
    package = importlib.import_module("chess_llm.training.training_args")

    assert legacy is package


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
    sys.modules.pop("chess_llm.training.training_args", None)

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
    sys.modules.pop("chess_llm.training.training_args", None)

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
    sys.modules.pop("chess_llm.training.training_args", None)

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


def test_training_step_estimate_accounts_for_accumulation_and_world_size():
    sys.modules.setdefault(
        "trl",
        types.SimpleNamespace(SFTConfig=type("FakeSFTConfig", (), {})),
    )
    sys.modules.pop("chess_llm.training.training_args", None)

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
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig))
    sys.modules.pop("chess_llm.training.training_args", None)

    from chess_llm.training.phases import PHASE_A
    from chess_llm.training.training_args import build_sft_config

    cfg = build_sft_config(
        PHASE_A,
        tmp_path,
        train_dataset_size=3200,
        num_train_epochs=1,
        world_size=2,
    )

    assert cfg.kwargs["num_train_epochs"] == 1
    assert cfg.kwargs["warmup_steps"] == 2


def test_build_sft_config_prefers_warmup_steps_when_dataset_size_known(
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
            max_steps=None,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig))
    sys.modules.pop("chess_llm.training.training_args", None)

    from chess_llm.training.phases import PHASE_A
    from chess_llm.training.training_args import build_sft_config

    cfg = build_sft_config(
        PHASE_A,
        tmp_path,
        train_dataset_size=128,
        max_steps=10,
    )

    assert cfg.kwargs["warmup_steps"] == 1
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
    sys.modules.pop("chess_llm.training.training_args", None)

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
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig))
    sys.modules.pop("chess_llm.training.training_args", None)

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
