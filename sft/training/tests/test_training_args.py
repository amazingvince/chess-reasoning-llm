"""Tests for SFTConfig construction without importing real TRL."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from config.phases import PHASE_A


class _FakeSFTConfig:
    def __init__(
        self,
        output_dir=None,
        run_name=None,
        packing=None,
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
        max_length=None,
        eos_token=None,
        assistant_only_loss=None,
    ):
        self.kwargs = dict(locals())
        self.kwargs.pop("self")


def test_sft_config_masks_prompt_tokens_and_enables_liger(monkeypatch, tmp_path: Path):
    fake_trl = types.SimpleNamespace(SFTConfig=_FakeSFTConfig)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    sys.modules.pop("config.training_args", None)

    training_args = importlib.import_module("config.training_args")
    importlib.reload(training_args)

    cfg = training_args.build_sft_config(PHASE_A, tmp_path)

    assert cfg.kwargs["assistant_only_loss"] is True
    assert cfg.kwargs["packing"] is False
    assert cfg.kwargs["use_liger_kernel"] is True
    assert cfg.kwargs["eval_steps"] == 5000
    assert cfg.kwargs["save_steps"] == 5000
    assert cfg.kwargs["save_safetensors"] is True
    assert cfg.kwargs["save_only_model"] is True


def test_sft_config_uses_qwen3_safe_liger_config(monkeypatch, tmp_path: Path):
    fake_trl = types.SimpleNamespace(SFTConfig=_FakeSFTConfig)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    sys.modules.pop("config.training_args", None)

    training_args = importlib.import_module("config.training_args")
    importlib.reload(training_args)

    cfg = training_args.build_sft_config(PHASE_A, tmp_path, model_type="qwen3")

    assert cfg.kwargs["use_liger_kernel"] is True
    assert cfg.kwargs["liger_kernel_config"] == {
        "fused_linear_cross_entropy": False,
    }


def test_sft_config_can_disable_trainer_eval(monkeypatch, tmp_path: Path):
    fake_trl = types.SimpleNamespace(SFTConfig=_FakeSFTConfig)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    sys.modules.pop("config.training_args", None)

    training_args = importlib.import_module("config.training_args")
    importlib.reload(training_args)

    cfg = training_args.build_sft_config(PHASE_A, tmp_path, trainer_eval=False)

    assert cfg.kwargs["eval_strategy"] == "no"
    assert cfg.kwargs["eval_steps"] is None
    assert cfg.kwargs["load_best_model_at_end"] is False
    assert cfg.kwargs["metric_for_best_model"] is None
