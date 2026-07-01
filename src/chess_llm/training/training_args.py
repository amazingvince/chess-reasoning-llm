"""Build SFTConfig from PhaseConfig for TRL SFTTrainer."""

from __future__ import annotations

import inspect
import logging
import math
import os
from pathlib import Path

from trl import SFTConfig

from chess_llm.training.phases import PhaseConfig

logger = logging.getLogger(__name__)

DEFAULT_TRAINER_EVAL_STEPS = 5000
DEFAULT_TRAINER_SAVE_STEPS = 5000
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_DATASET_NUM_PROC = min(8, max(1, os.cpu_count() or 1))
FLASH_ATTENTION_VARIANTS = {
    "flash_attention_2",
    "flash_attention_3",
    "flash_attention_4",
    "hf_flash_attention_2",
    "kernels-community/flash-attn2",
    "kernels-community/flash-attn3",
    "kernels-community/flash-attn4",
    "kernels-community/vllm-flash-attn3",
}
PACKING_CHOICES = ("auto", "on", "off")


def _is_qwen3_family(model_type: str | None) -> bool:
    return bool(model_type and model_type.startswith("qwen3"))


def _filter_supported_sft_kwargs(config_kwargs: dict) -> dict:
    """Keep only kwargs accepted by the installed TRL ``SFTConfig``."""
    supported = set(inspect.signature(SFTConfig.__init__).parameters)
    return {key: value for key, value in config_kwargs.items() if key in supported}


def _supports_flash_attention_packing(attn_implementation: str | None) -> bool:
    if not attn_implementation:
        return False
    base_implementation = attn_implementation.split("@", 1)[0]
    return bool(base_implementation in FLASH_ATTENTION_VARIANTS)


def resolve_packing_settings(
    attn_implementation: str | None,
    packing: str = "auto",
) -> tuple[bool, bool]:
    """Return ``(packing, padding_free)`` for the requested train settings."""
    if packing not in PACKING_CHOICES:
        raise ValueError(f"Unknown packing mode {packing!r}; expected one of {PACKING_CHOICES}")
    flash_packing_supported = _supports_flash_attention_packing(attn_implementation)
    if packing == "off":
        return False, False
    if packing == "on":
        return True, flash_packing_supported
    return flash_packing_supported, flash_packing_supported


def _default_world_size() -> int:
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return 1


def estimate_training_steps(
    phase: PhaseConfig,
    *,
    train_dataset_size: int,
    num_train_epochs: float | None = None,
    max_steps: int | None = None,
    per_device_train_batch_size: int = PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    world_size: int | None = None,
) -> int:
    """Estimate optimizer update steps for warmup scheduling."""
    if max_steps is not None and max_steps > 0:
        return max_steps
    if train_dataset_size <= 0:
        return 0

    resolved_world_size = _default_world_size() if world_size is None else max(1, world_size)
    effective_batch = per_device_train_batch_size * resolved_world_size
    batches_per_epoch = math.ceil(train_dataset_size / effective_batch)
    updates_per_epoch = max(1, math.ceil(batches_per_epoch / gradient_accumulation_steps))
    effective_epochs = phase.epochs if num_train_epochs is None else num_train_epochs
    return max(1, math.ceil(updates_per_epoch * effective_epochs))


def _warmup_steps_from_ratio(ratio: float, total_steps: int) -> int:
    if ratio <= 0 or total_steps <= 0:
        return 0
    return max(1, math.ceil(total_steps * ratio))


def build_sft_config(
    phase: PhaseConfig,
    output_dir: Path,
    run_name: str | None = None,
    report_to: str | list[str] = "wandb",
    model_type: str | None = None,
    use_liger_kernel: bool = True,
    num_train_epochs: float | None = None,
    max_steps: int | None = None,
    eval_steps: int | None = None,
    save_steps: int | None = None,
    logging_steps: int | None = None,
    trainer_eval: bool = True,
    train_dataset_size: int | None = None,
    world_size: int | None = None,
    attn_implementation: str | None = None,
    liger_fused_linear_cross_entropy: bool = False,
    gradient_checkpointing: bool = False,
    packing: str = "auto",
    max_length: int = 2048,
) -> SFTConfig:
    """Build a TRL SFTConfig for the given phase.

    Key optimizations:
    - Liger kernels with Qwen3.5 fused linear CE disabled by default
    - Assistant-only loss masking so prompts are not learned as targets
    - bf16 + TF32 matmul on CUDA hardware

    Qwen-specific:
    - eos_token set to ``<|im_end|>`` to align with the Qwen chat
      template's turn terminator (per TRL docs for Qwen models).

    Flash Attention is enabled at model load time; the selected backend is used
    here only to decide whether TRL packing can safely use padding-free batches.
    """
    if run_name is None:
        run_name = f"chess-sft-phase-{phase.name}"
    effective_num_train_epochs = phase.epochs if num_train_epochs is None else num_train_epochs
    signature = inspect.signature(SFTConfig.__init__).parameters

    if "assistant_only_loss" not in signature:
        raise RuntimeError(
            "Installed TRL SFTConfig does not support assistant_only_loss. "
            "Refusing to train with prompt tokens included in the loss."
        )

    is_qwen3_family = _is_qwen3_family(model_type)
    packing_enabled, padding_free_enabled = resolve_packing_settings(
        attn_implementation,
        packing=packing,
    )

    if use_liger_kernel and is_qwen3_family and "liger_kernel_config" not in signature:
        logger.warning(
            "Disabling Liger kernels for Qwen3-family model because the installed TRL "
            "SFTConfig cannot disable fused_linear_cross_entropy."
        )
        use_liger_kernel = False

    effective_eval_steps = DEFAULT_TRAINER_EVAL_STEPS if eval_steps is None else eval_steps
    save_without_eval = not trainer_eval and save_steps is not None
    effective_save_steps = (
        DEFAULT_TRAINER_SAVE_STEPS if save_steps is None else save_steps
    )
    effective_save_strategy = "steps" if trainer_eval or save_without_eval else "no"
    warmup_steps = None
    if train_dataset_size is not None:
        warmup_total_steps = estimate_training_steps(
            phase,
            train_dataset_size=train_dataset_size,
            num_train_epochs=effective_num_train_epochs,
            max_steps=max_steps,
            world_size=world_size,
        )
        warmup_steps = _warmup_steps_from_ratio(
            phase.warmup_ratio,
            warmup_total_steps,
        )

    config_kwargs = dict(
        output_dir=str(output_dir),
        run_name=run_name,
        packing=packing_enabled,
        assistant_only_loss=True,
        # Training
        num_train_epochs=effective_num_train_epochs,
        learning_rate=phase.learning_rate,
        weight_decay=phase.weight_decay,
        lr_scheduler_type="cosine",
        # Batch
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        # Precision + memory
        bf16=True,
        tf32=True,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Liger kernel
        use_liger_kernel=use_liger_kernel,
        # Evaluation/checkpointing. Benchmark eval remains the quality gate;
        # trainer eval is a bounded loss probe for checkpoint selection.
        eval_strategy="steps" if trainer_eval else "no",
        eval_steps=effective_eval_steps if trainer_eval else None,
        save_strategy=effective_save_strategy,
        save_steps=effective_save_steps if trainer_eval or save_without_eval else None,
        save_total_limit=3 if trainer_eval or save_without_eval else None,
        load_best_model_at_end=trainer_eval,
        metric_for_best_model="eval_loss" if trainer_eval else None,
        greater_is_better=False if trainer_eval else None,
        save_safetensors=True,
        # Keep optimizer/scheduler/RNG state in step checkpoints so interrupted
        # long runs can resume from the periodic recovery checkpoints.
        save_only_model=False,
        # Logging
        logging_steps=50 if logging_steps is None else logging_steps,
        report_to=report_to,
        # Misc
        dataloader_num_workers=4,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        seed=42,
        dataloader_pin_memory=True,
        dataset_num_proc=DEFAULT_DATASET_NUM_PROC,
        include_num_input_tokens_seen=True,
        pad_to_multiple_of=8,
        padding_free=padding_free_enabled,
    )

    if packing_enabled:
        config_kwargs["packing_strategy"] = "bfd"

    # TRL has renamed a few SFTConfig fields across versions.
    if "max_seq_length" in signature:
        config_kwargs["max_seq_length"] = max_length
    elif "max_length" in signature:
        config_kwargs["max_length"] = max_length

    if "eos_token" in signature:
        config_kwargs["eos_token"] = "<|im_end|>"

    if "use_liger_kernel" not in signature and "use_liger" in signature:
        config_kwargs.pop("use_liger_kernel", None)
        config_kwargs["use_liger"] = use_liger_kernel

    if use_liger_kernel and is_qwen3_family and "liger_kernel_config" in signature:
        config_kwargs["liger_kernel_config"] = {
            "cross_entropy": False,
            "fused_linear_cross_entropy": liger_fused_linear_cross_entropy,
        }
        logger.info(
            "Using Qwen3.5 Liger config: cross_entropy=False, "
            "fused_linear_cross_entropy=%s",
            liger_fused_linear_cross_entropy,
        )

    if max_steps is not None:
        config_kwargs["max_steps"] = max_steps

    if warmup_steps is not None and "warmup_steps" in signature:
        config_kwargs["warmup_steps"] = warmup_steps
    else:
        config_kwargs["warmup_ratio"] = phase.warmup_ratio

    return SFTConfig(**_filter_supported_sft_kwargs(config_kwargs))
