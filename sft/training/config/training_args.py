"""Build SFTConfig from PhaseConfig for TRL SFTTrainer."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

from trl import SFTConfig

from config.phases import PhaseConfig

logger = logging.getLogger(__name__)

DEFAULT_TRAINER_EVAL_STEPS = 5000
DEFAULT_TRAINER_SAVE_STEPS = 5000


def _filter_supported_sft_kwargs(config_kwargs: dict) -> dict:
    """Keep only kwargs accepted by the installed TRL ``SFTConfig``."""
    supported = set(inspect.signature(SFTConfig.__init__).parameters)
    return {key: value for key, value in config_kwargs.items() if key in supported}


def build_sft_config(
    phase: PhaseConfig,
    output_dir: Path,
    run_name: str | None = None,
    report_to: str | list[str] = "wandb",
    model_type: str | None = None,
    use_liger_kernel: bool = True,
    max_steps: int | None = None,
    eval_steps: int | None = None,
    save_steps: int | None = None,
    logging_steps: int | None = None,
    trainer_eval: bool = True,
) -> SFTConfig:
    """Build a TRL SFTConfig for the given phase.

    Key optimizations:
    - Liger kernels (20% throughput, 60% less memory)
    - Assistant-only loss masking so prompts are not learned as targets
    - bf16 + gradient checkpointing

    Qwen-specific:
    - eos_token set to ``<|im_end|>`` to align with the Qwen chat
      template's turn terminator (per TRL docs for Qwen models).

    Flash Attention 2 is enabled at model load time, not here.
    """
    if run_name is None:
        run_name = f"chess-sft-phase-{phase.name}"
    signature = inspect.signature(SFTConfig.__init__).parameters

    if "assistant_only_loss" not in signature:
        raise RuntimeError(
            "Installed TRL SFTConfig does not support assistant_only_loss. "
            "Refusing to train with prompt tokens included in the loss."
        )

    if use_liger_kernel and model_type == "qwen3" and "liger_kernel_config" not in signature:
        logger.warning(
            "Disabling Liger kernels for Qwen3 because the installed TRL "
            "SFTConfig cannot disable fused_linear_cross_entropy."
        )
        use_liger_kernel = False

    effective_eval_steps = DEFAULT_TRAINER_EVAL_STEPS if eval_steps is None else eval_steps
    effective_save_steps = DEFAULT_TRAINER_SAVE_STEPS if save_steps is None else save_steps

    config_kwargs = dict(
        output_dir=str(output_dir),
        run_name=run_name,
        packing=False,
        assistant_only_loss=True,
        # Training
        num_train_epochs=phase.epochs,
        learning_rate=phase.learning_rate,
        warmup_ratio=phase.warmup_ratio,
        weight_decay=phase.weight_decay,
        lr_scheduler_type="cosine",
        # Batch
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,  # effective batch = 32
        # Precision + memory
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Liger kernel
        use_liger_kernel=use_liger_kernel,
        # Evaluation/checkpointing. Benchmark eval remains the quality gate;
        # trainer eval is a bounded loss probe for checkpoint selection.
        eval_strategy="steps" if trainer_eval else "no",
        eval_steps=effective_eval_steps if trainer_eval else None,
        save_strategy="steps",
        save_steps=effective_save_steps,
        save_total_limit=3,
        load_best_model_at_end=trainer_eval,
        metric_for_best_model="eval_loss" if trainer_eval else None,
        greater_is_better=False if trainer_eval else None,
        save_safetensors=True,
        save_only_model=True,
        # Logging
        logging_steps=50 if logging_steps is None else logging_steps,
        report_to=report_to,
        # Misc
        dataloader_num_workers=4,
        seed=42,
        dataloader_pin_memory=True,
    )

    # TRL has renamed a few SFTConfig fields across versions.
    if "max_seq_length" in signature:
        config_kwargs["max_seq_length"] = 2048
    elif "max_length" in signature:
        config_kwargs["max_length"] = 2048

    if "eos_token" in signature:
        config_kwargs["eos_token"] = "<|im_end|>"

    if "use_liger_kernel" not in signature and "use_liger" in signature:
        config_kwargs.pop("use_liger_kernel", None)
        config_kwargs["use_liger"] = use_liger_kernel

    if use_liger_kernel and model_type == "qwen3" and "liger_kernel_config" in signature:
        config_kwargs["liger_kernel_config"] = {
            "fused_linear_cross_entropy": False,
        }
        logger.info(
            "Using Qwen3-safe Liger config: fused_linear_cross_entropy=False"
        )

    if max_steps is not None:
        config_kwargs["max_steps"] = max_steps

    return SFTConfig(**_filter_supported_sft_kwargs(config_kwargs))
