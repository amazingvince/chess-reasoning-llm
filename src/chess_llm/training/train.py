#!/usr/bin/env python3
"""Main CLI for SFT training.

Usage:
    chess-llm-train --phase a                        # Train Phase A
    chess-llm-train --phase b                        # Train Phase B from A checkpoint
    chess-llm-train --phase c                        # Train Phase C from B checkpoint
    chess-llm-train --phase schedule                 # One long run, time-varying tier mix
    chess-llm-train --phase a --eval-only            # Baseline eval only (no training)
    chess-llm-train --phase a --dry-run              # Print config + data summary, exit
    chess-llm-train --phase a --smoke-run            # Short end-to-end smoke test
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_llm.training.model_loading import (
    ATTENTION_IMPLEMENTATION_CHOICES,
    attention_candidates,
)
from chess_llm.training.eval_exit_codes import EVAL_INFRA_FAILURE_EXIT_CODE
from chess_llm.training.logging_utils import configure_cli_logging
from chess_llm.training.phase_gate import is_count_metric
from chess_llm.training.wandb_utils import wandb_config_error

# Default paths (override with CLI args or env vars)
DEFAULT_CHESS_SFT_ROOT = Path(os.environ.get("CHESS_SFT_OUTPUT", "chess_sft_data"))
DEFAULT_DATA_ROOT = DEFAULT_CHESS_SFT_ROOT / "output"
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("CHESS_SFT_CHECKPOINTS", "chess_sft_checkpoints"))
DEFAULT_BENCHMARK_DIR = DEFAULT_CHESS_SFT_ROOT / "benchmark"
DEFAULT_STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish") or "stockfish"

DEFAULT_MAX_EVAL_EXAMPLES = 2048
SMOKE_MAX_TRAIN_EXAMPLES = 128
SMOKE_MAX_EVAL_EXAMPLES = 64
SMOKE_MAX_BENCHMARK_EXAMPLES_PER_SPLIT = 32
SMOKE_MAX_STEPS = 10
SHORT_RUN_CADENCE_MAX_STEPS = 500
PACKING_CHOICES = ("auto", "on", "off")

configure_cli_logging()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunOverrides:
    """Effective runtime limits after applying smoke-mode defaults."""

    max_train_examples: int | None = None
    max_eval_examples: int | None = None
    max_benchmark_examples_per_split: int | None = None
    num_train_epochs: float | None = None
    max_steps: int | None = None
    eval_steps: int | None = None
    save_steps: int | None = None
    logging_steps: int | None = None
    skip_trainer_eval: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chess SFT training harness")
    parser.add_argument(
        "--phase", required=True, choices=["a", "b", "c", "schedule"],
        help=(
            "Training phase (a=Foundation, b=Understanding, c=Planning, "
            "schedule=one long run over all tiers with a time-varying mix)"
        ),
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help=f"Root directory with tier{{N}}/ subdirectories (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help=f"Root for checkpoints (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR,
        help=f"Frozen benchmark directory (default: {DEFAULT_BENCHMARK_DIR})",
    )
    parser.add_argument(
        "--run-ledger",
        type=Path,
        default=None,
        help="Append post-training/eval-only evaluation metadata to this JSONL ledger.",
    )
    parser.add_argument(
        "--artifact-mirror-dir",
        type=Path,
        default=None,
        help="Mirror evaluation artifacts under this directory by eval run id.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help=(
            "Base Hugging Face model ID for Phase A. Overrides "
            "CHESS_SFT_BASE_MODEL for this process."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config and data summary, then exit")
    parser.add_argument(
        "--smoke-run",
        action="store_true",
        help="Run a bounded end-to-end smoke test with tiny train/eval subsets and short benchmark eval",
    )
    parser.add_argument("--eval-only", action="store_true", help="Run benchmark eval only (no training)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-training benchmark eval")
    parser.add_argument(
        "--require-phase-gate",
        action="store_true",
        help="Make benchmark failures fatal and require prior phases to have PASSED sentinels",
    )
    parser.add_argument(
        "--max-train-examples", type=int, default=None,
        help="Limit mixed training dataset size after sampling/upsampling (useful for smoke tests)",
    )
    parser.add_argument(
        "--task-upsample",
        action="append",
        default=[],
        metavar="TASK=FACTOR",
        help=(
            "Upsample a specific task on the train split before any max-train cap. "
            "Repeat for multiple tasks, e.g. --task-upsample 1.5_state_tracking=8"
        ),
    )
    parser.add_argument(
        "--task-include",
        action="append",
        default=[],
        metavar="TASK",
        help=(
            "Keep only examples from this task before training sanitization. "
            "Repeat for multiple tasks."
        ),
    )
    parser.add_argument(
        "--task-exclude",
        action="append",
        default=[],
        metavar="TASK",
        help=(
            "Drop examples from this task before training sanitization. "
            "Repeat for multiple tasks."
        ),
    )
    parser.add_argument(
        "--move-only-task",
        action="append",
        default=[],
        metavar="TASK",
        help=(
            "Rewrite this task's assistant target to a compact "
            "<think>...</think><move>...</move> move-only answer. "
            "Repeat for multiple tasks."
        ),
    )
    parser.add_argument(
        "--schedule-total-examples", type=int, default=None,
        help=(
            "Total training-example budget for --phase schedule "
            "(default: sum of tier train pools; further capped by "
            "--max-train-examples). Ignored for phases a/b/c."
        ),
    )
    parser.add_argument(
        "--max-eval-examples", type=int, default=None,
        help=(
            "Limit trainer eval dataset size after mixing "
            f"(default: {DEFAULT_MAX_EVAL_EXAMPLES}; use 0 for the full eval set)"
        ),
    )
    parser.add_argument(
        "--max-benchmark-examples-per-split", type=int, default=None,
        help="Limit benchmark examples per split during evaluation",
    )
    parser.add_argument(
        "--full-benchmark-eval",
        "--full-benchmark",
        dest="full_benchmark_eval",
        action="store_true",
        help="Evaluate every benchmark split instead of phase-aware default splits.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help=(
            "Override trainer max_steps; runs of <= "
            f"{SHORT_RUN_CADENCE_MAX_STEPS} steps also tighten eval/save cadence"
        ),
    )
    parser.add_argument(
        "--num-train-epochs",
        type=_positive_float,
        default=None,
        help="Override trainer num_train_epochs; use 1 for a one-pass generated-data run",
    )
    parser.add_argument(
        "--learning-rate",
        type=_positive_float,
        default=None,
        help="Override the phase learning rate for controlled continuation probes",
    )
    parser.add_argument(
        "--trainer-eval-steps",
        type=int,
        default=None,
        help="Trainer loss-eval cadence in optimizer steps (default: 5000 for full runs)",
    )
    parser.add_argument(
        "--trainer-save-steps",
        type=int,
        default=None,
        help="Checkpoint save cadence in optimizer steps (default: same as trainer eval cadence)",
    )
    parser.add_argument(
        "--skip-trainer-eval",
        action="store_true",
        help="Disable in-training loss eval and save the final checkpoint as best/",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH|auto",
        help=(
            "Resume Trainer state from a checkpoint path. Pass without a value, "
            "or pass 'auto', to use the latest checkpoint-N directory under "
            "the phase output directory."
        ),
    )
    parser.add_argument("--wandb-project", type=str, default="chess-sft", help="W&B project name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B reporting (useful for CI)")
    parser.add_argument(
        "--allow-wandb-offline",
        action="store_true",
        help=(
            "Allow WANDB_MODE=offline/dryrun for an intentional offline W&B run. "
            "Smoke runs allow this automatically."
        ),
    )
    parser.add_argument("--run-name", type=str, default=None, help="W&B run name override")
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional W&B group for training + eval runs")
    parser.add_argument(
        "--disable-liger-kernel",
        action="store_true",
        help="Disable Liger kernels for debugging or benchmarking",
    )
    parser.add_argument(
        "--enable-liger-fused-linear-ce",
        action="store_true",
        help=(
            "Enable Liger fused linear cross entropy while keeping other Liger "
            "kernels enabled. Off by default for the current Qwen3.5 runtime "
            "because local probes show it is slower than the standard loss path."
        ),
    )
    parser.add_argument(
        "--disable-liger-fused-linear-ce",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "Enable activation checkpointing. This saves memory but slows the "
            "default Qwen/Qwen3.5-0.8B training loop."
        ),
    )
    parser.add_argument(
        "--inference-backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Generation backend for benchmark evaluation",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=ATTENTION_IMPLEMENTATION_CHOICES,
        default="auto",
        help="Attention backend for training and transformers eval model loads",
    )
    parser.add_argument(
        "--packing",
        choices=PACKING_CHOICES,
        default="auto",
        help=(
            "SFT sequence packing mode. auto enables packing only for selected "
            "flash-attention backends; on forces standard packing for any "
            "attention backend; off disables packing."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=_positive_int,
        default=2048,
        help="Maximum tokenized sequence length for SFT training examples.",
    )
    parser.add_argument(
        "--pure-bf16",
        action="store_true",
        help=(
            "Load training weights in the checkpoint dtype (bf16) instead of "
            "fp32 master weights. Saves memory, but small Adam updates at "
            "lr<=2e-5 can round to zero without fp32 master weights."
        ),
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Benchmark generation batch size for transformers eval (default: 16)",
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=512,
        help="Benchmark generation max_new_tokens (default: 512)",
    )
    parser.add_argument(
        "--eval-acpl-depth",
        type=int,
        default=20,
        help="Stockfish depth for benchmark ACPL when enabled (default: 20)",
    )
    parser.add_argument(
        "--no-acpl",
        action="store_true",
        help="Disable benchmark ACPL. Phase C post-eval will fail its ACPL gate.",
    )
    parser.add_argument(
        "--full-acpl-report",
        action="store_true",
        help="Compute ACPL for every split instead of only Phase C planning gate examples.",
    )
    parser.add_argument(
        "--stockfish-path", type=str,
        default=DEFAULT_STOCKFISH_PATH,
        help="Path to Stockfish binary for ACPL (default: from STOCKFISH_PATH env)",
    )
    return parser.parse_args()


def _register_trl_flash_attention_variant(attn_implementation: str | None) -> None:
    """Teach TRL about pinned HF flash-attention kernel names before packing checks run.

    Only genuine FlashAttention backends are registered. Non-flash backends
    (sdpa/eager) are left unregistered so TRL's packing cross-contamination
    warning stays loud for them.
    """
    if not attn_implementation:
        return
    from chess_llm.training.training_args import is_flash_attention_implementation

    if not is_flash_attention_implementation(attn_implementation):
        return
    try:
        sft_trainer_module = importlib.import_module("trl.trainer.sft_trainer")
    except Exception:
        return
    variants = getattr(sft_trainer_module, "FLASH_ATTENTION_VARIANTS", None)
    if isinstance(variants, set):
        variants.add(attn_implementation)


def _training_torch_dtype(pure_bf16: bool) -> str:
    """Weight-load dtype for full fine-tunes.

    Default is fp32 master weights (bf16=True then autocasts compute);
    loading in bf16 directly makes Adam updates at lr<=2e-5 round to zero.
    Eval-only paths do not use this — they keep loading with 'auto'.
    """
    return "auto" if pure_bf16 else "float32"


def _training_launcher_preflight_error(torch_module: Any | None = None) -> str | None:
    """Return an error when multiple GPUs are visible outside torchrun."""
    if torch_module is None:
        try:
            import torch as torch_module
        except Exception:
            return None

    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None) if cuda is not None else None
    device_count_fn = getattr(cuda, "device_count", None) if cuda is not None else None
    if not callable(is_available) or not callable(device_count_fn) or not is_available():
        return None

    device_count = int(device_count_fn())
    if device_count <= 1:
        return None

    world_size_text = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(world_size_text)
    except ValueError:
        world_size = 1
    local_rank = os.environ.get("LOCAL_RANK")
    if world_size > 1 and local_rank is not None:
        return None

    return (
        f"{device_count} CUDA devices are visible, but this process was not "
        "launched by torchrun. Refusing single-process multi-GPU training "
        "because it can fall into DataParallel and crash or silently diverge "
        "under flash-attention kernels. Relaunch with "
        f"`torchrun --nproc_per_node={device_count} -m chess_llm.training.train ...` "
        "or set CUDA_VISIBLE_DEVICES to a single GPU."
    )


def _float_metric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trainer_num_tokens_seen(trainer: Any) -> float | None:
    state = getattr(trainer, "state", None)
    num_tokens = _float_metric(getattr(state, "num_input_tokens_seen", None))
    if num_tokens is not None:
        return num_tokens

    log_history = getattr(state, "log_history", None) or []
    for entry in reversed(log_history):
        if not isinstance(entry, dict) or "num_tokens" not in entry:
            continue
        num_tokens = _float_metric(entry["num_tokens"])
        if num_tokens is not None:
            return num_tokens
    return None


def _augment_train_metrics_with_token_throughput(
    trainer: Any,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Add normalized token throughput to Trainer train metrics when available."""
    augmented = dict(metrics)
    runtime = _float_metric(augmented.get("train_runtime"))
    num_tokens = _trainer_num_tokens_seen(trainer)
    if runtime is None or runtime <= 0 or num_tokens is None or num_tokens <= 0:
        return augmented

    augmented["train_num_tokens"] = int(num_tokens) if num_tokens.is_integer() else num_tokens
    augmented["train_tokens_per_second"] = num_tokens / runtime
    return augmented


def main() -> int:
    args = parse_args()
    if getattr(args, "base_model", None):
        os.environ["CHESS_SFT_BASE_MODEL"] = args.base_model
    overrides = _resolve_run_overrides(args)
    task_upsample = _parse_task_upsample_overrides(
        getattr(args, "task_upsample", [])
    )
    data_transform_config = _build_data_transform_config(args)

    from chess_llm.training.data.mixer import build_phase_dataset, summarize_phase_data
    from chess_llm.training.phases import PHASES, resolve_checkpoint

    phase = PHASES.get(args.phase)
    is_schedule = phase is None
    if is_schedule:
        from chess_llm.training.schedule import SCHEDULES, validate_schedule

        phase = SCHEDULES[args.phase]
        validate_schedule(phase)
        schedule_error = _schedule_mode_config_error(args)
        if schedule_error is not None:
            logger.error(schedule_error)
            return 2
        if not args.no_wandb and (args.run_name is None or args.wandb_group is None):
            from datetime import datetime

            default_run_id = f"schedule-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            if args.run_name is None:
                args.run_name = default_run_id
            if args.wandb_group is None:
                args.wandb_group = default_run_id
            logger.info("Schedule W&B run id: %s", default_run_id)

    # Schedule runs measure against the full benchmark without hard gates.
    eval_phase = None if is_schedule else phase.name
    eval_full_benchmark = (
        True if is_schedule else getattr(args, "full_benchmark_eval", False)
    )
    eval_soft_gate = True if is_schedule else not args.require_phase_gate

    output_dir = args.output_root / f"phase_{phase.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(phase.display_name)
    logger.info("=" * 60)

    # Resolve starting checkpoint (for training)
    model_path = resolve_checkpoint(
        phase,
        args.output_root,
        require_passed=args.require_phase_gate,
    )
    logger.info("Model: %s", model_path)
    logger.info("Data root: %s", args.data_root)
    logger.info("Output: %s", output_dir)

    # --- Dry run ---
    if args.dry_run:
        if is_schedule:
            return _schedule_dry_run(
                phase,
                args,
                overrides,
                task_upsample,
                data_transform_config,
            )
        logger.info("--- DRY RUN: data summary ---")
        summary = summarize_phase_data(
            phase,
            args.data_root,
            task_upsample=task_upsample,
            **_data_transform_kwargs(data_transform_config),
        )
        for key, count in sorted(summary.items()):
            logger.info("  %-12s %8d", key, count)
        train_ds, eval_ds = build_phase_dataset(
            phase,
            args.data_root,
            task_upsample=task_upsample,
            **_data_transform_kwargs(data_transform_config),
        )
        train_split_count = len(train_ds)
        eval_split_count = len(eval_ds)
        effective_train_count = (
            min(train_split_count, overrides.max_train_examples)
            if overrides.max_train_examples is not None
            else train_split_count
        )
        effective_trainer_eval_count = (
            0
            if overrides.skip_trainer_eval
            else (
                min(eval_split_count, overrides.max_eval_examples)
                if overrides.max_eval_examples is not None
                else eval_split_count
            )
        )
        if effective_train_count != train_split_count:
            logger.info(
                "Train split: %d examples (%d after cap)",
                train_split_count,
                effective_train_count,
            )
        else:
            logger.info("Train split: %d examples", train_split_count)
        if overrides.skip_trainer_eval:
            logger.info(
                "Eval split: %d examples (trainer eval disabled)",
                eval_split_count,
            )
        else:
            logger.info(
                "Eval split: %d examples (%d used by trainer)",
                eval_split_count,
                effective_trainer_eval_count,
            )
        learning_rate_override = getattr(args, "learning_rate", None)
        effective_learning_rate = (
            learning_rate_override if learning_rate_override is not None else phase.learning_rate
        )
        logger.info("Learning rate: %s", effective_learning_rate)
        effective_epochs = overrides.num_train_epochs or phase.epochs
        logger.info("Epochs: %s", _format_epoch_count(effective_epochs))
        from chess_llm.training.training_args import resolve_packing_settings

        # Resolve the attention backend the same way the real run will
        # (first load candidate), so the packing report matches training.
        dry_run_attn = attention_candidates(args.attn_implementation)[0]
        dry_run_packing, dry_run_padding_free = resolve_packing_settings(
            dry_run_attn,
            packing=getattr(args, "packing", "auto"),
        )
        logger.info(
            "Packing: %s (mode=%s, padding_free=%s, attn=%s), max_length: %d",
            dry_run_packing,
            getattr(args, "packing", "auto"),
            dry_run_padding_free,
            dry_run_attn or "<default>",
            getattr(args, "max_length", 2048),
        )
        logger.info("Effective batch size: 32 (4 * 8 accumulation)")
        logger.info("Eval backend: %s", args.inference_backend)
        for detail in _build_dry_run_training_details(
            phase,
            args,
            overrides,
            train_dataset_size=effective_train_count,
        ):
            logger.info(detail)
        if args.smoke_run or overrides.max_steps is not None or overrides.num_train_epochs is not None:
            logger.info(
                "Runtime overrides: train<=%s eval<=%s benchmark/split<=%s epochs=%s max_steps=%s",
                overrides.max_train_examples,
                overrides.max_eval_examples,
                overrides.max_benchmark_examples_per_split,
                (
                    _format_epoch_count(overrides.num_train_epochs)
                    if overrides.num_train_epochs is not None
                    else None
                ),
                overrides.max_steps,
            )
        return 0

    allow_wandb_offline = args.allow_wandb_offline or args.smoke_run
    wandb_error = wandb_config_error(
        no_wandb=args.no_wandb,
        allow_offline=allow_wandb_offline,
    )
    if wandb_error is not None:
        logger.error(wandb_error)
        return EVAL_INFRA_FAILURE_EXIT_CODE

    if not args.eval_only:
        launcher_error = _training_launcher_preflight_error()
        if launcher_error is not None:
            logger.error(launcher_error)
            return EVAL_INFRA_FAILURE_EXIT_CODE

    if not args.eval_only and not args.skip_eval:
        preflight_error = _post_training_eval_preflight_error(
            args.benchmark_dir,
            phase=eval_phase,
            full_benchmark=eval_full_benchmark,
            inference_backend=args.inference_backend,
        )
        if preflight_error is not None:
            logger.error("Post-training benchmark eval preflight failed: %s", preflight_error)
            return EVAL_INFRA_FAILURE_EXIT_CODE

    # --- Eval only ---
    if args.eval_only:
        # Evaluate this phase's own best/ checkpoint if it exists,
        # otherwise fall back to the starting checkpoint (base model / prev phase).
        best_dir = output_dir / "best"
        if best_dir.exists():
            eval_model = str(best_dir)
            pred_path = output_dir / "eval_predictions.jsonl"
        else:
            # Untrained phase: write to a separate artifact so fallback-model
            # metrics never land in eval_predictions.results.json, which
            # _find_best_historical_baseline merges as this phase's baseline.
            eval_model = model_path
            pred_path = output_dir / "eval_only_predictions.jsonl"
        logger.info("Eval-only model: %s", eval_model)

        baseline_path = _find_best_historical_baseline(args.output_root, phase.name)
        eval_rc = _run_eval(
            eval_model, args.benchmark_dir,
            pred_path,
            baseline_path=baseline_path,
            phase=eval_phase,
            pass_k=8 if phase.name == "c" else 1,
            stockfish_path=args.stockfish_path,
            inference_backend=args.inference_backend,
            attn_implementation=args.attn_implementation,
            eval_batch_size=args.eval_batch_size,
            eval_max_new_tokens=args.eval_max_new_tokens,
            eval_acpl_depth=args.eval_acpl_depth,
            no_acpl=args.no_acpl,
            full_acpl_report=args.full_acpl_report,
            run_ledger=getattr(args, "run_ledger", None),
            artifact_mirror_dir=getattr(args, "artifact_mirror_dir", None),
            soft_gate=eval_soft_gate,
            max_examples_per_split=overrides.max_benchmark_examples_per_split,
            full_benchmark=eval_full_benchmark,
            wandb_project=args.wandb_project,
            wandb_run_name=args.run_name or f"phase-{phase.name}-eval-only",
            wandb_group=args.wandb_group,
            no_wandb=args.no_wandb,
            allow_wandb_offline=allow_wandb_offline,
        )
        if eval_rc == 0:
            _warn_if_soft_gate_failures(
                pred_path,
                require_phase_gate=args.require_phase_gate,
            )
        if eval_rc == 0 and best_dir.exists():
            from chess_llm.training.phases import PHASE_PASSED_SENTINEL, PHASE_READY_SENTINEL

            ready_sentinel = output_dir / PHASE_READY_SENTINEL
            ready_sentinel.write_text(
                f"Phase {phase.name} checkpoint ready after eval-only run.\n",
                encoding="utf-8",
            )
            logger.info("Wrote phase-ready sentinel: %s", ready_sentinel)
            if args.require_phase_gate:
                passed_sentinel = output_dir / PHASE_PASSED_SENTINEL
                passed_sentinel.write_text(
                    f"Phase {phase.name} passed eval-only benchmark checks.\n",
                    encoding="utf-8",
                )
                logger.info("Wrote phase gate sentinel: %s", passed_sentinel)
        elif eval_rc == 0 and args.require_phase_gate:
            logger.warning(
                "Phase %s eval-only succeeded for %s, but %s does not exist; "
                "not writing strict phase sentinels.",
                phase.name,
                eval_model,
                best_dir,
            )
        return eval_rc

    # --- Build datasets ---
    logger.info("Building datasets...")
    schedule_plans = None
    schedule_boundaries = None
    if is_schedule:
        # The example budget flows into build_schedule_dataset; the train
        # split must never pass through _limit_dataset, which would shuffle
        # away the planned segment order.
        train_ds, eval_ds, schedule_plans, schedule_boundaries = (
            _build_schedule_training_data(
                phase,
                args,
                overrides,
                task_upsample,
                output_dir,
                data_transform_config=data_transform_config,
            )
        )
    else:
        train_ds, eval_ds = build_phase_dataset(
            phase,
            args.data_root,
            task_upsample=task_upsample,
            **_data_transform_kwargs(data_transform_config),
        )
        train_ds = _limit_dataset(train_ds, overrides.max_train_examples, seed=42)
    eval_ds = _limit_dataset(eval_ds, overrides.max_eval_examples, seed=42)
    trainer_eval_ds, trainer_eval_enabled = _resolve_trainer_eval_dataset(
        eval_ds,
        skip_trainer_eval=overrides.skip_trainer_eval,
    )
    if trainer_eval_ds is None:
        logger.info("Train: %d examples, trainer eval disabled", len(train_ds))
    else:
        logger.info("Train: %d examples, Eval: %d examples", len(train_ds), len(eval_ds))

    # --- Load model + tokenizer ---
    logger.info("Loading model and tokenizer from %s", model_path)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token: %s", tokenizer.pad_token)

    training_torch_dtype = _training_torch_dtype(getattr(args, "pure_bf16", False))
    logger.info("Training weight load dtype: %s", training_torch_dtype)
    model_kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype": training_torch_dtype,
    }
    from chess_llm.training.model_loading import load_causal_lm_with_attention

    model, selected_attn = load_causal_lm_with_attention(
        AutoModelForCausalLM,
        model_path,
        model_kwargs,
        requested_attn=args.attn_implementation,
        logger=logger,
    )
    logger.info("Selected attention implementation: %s", selected_attn or "<default>")
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    logger.info("Model type: %s", model_type or "<unknown>")

    # --- Build SFTConfig ---
    from chess_llm.training.training_args import build_sft_config

    report_to: str | list[str] = "none" if args.no_wandb else "wandb"
    _configure_wandb_env(
        no_wandb=args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
    )
    sft_config = build_sft_config(
        phase,
        output_dir,
        run_name=args.run_name,
        report_to=report_to,
        model_type=model_type,
        use_liger_kernel=not args.disable_liger_kernel,
        num_train_epochs=overrides.num_train_epochs,
        liger_fused_linear_cross_entropy=(
            getattr(args, "enable_liger_fused_linear_ce", False)
            and not getattr(args, "disable_liger_fused_linear_ce", False)
        ),
        gradient_checkpointing=getattr(args, "gradient_checkpointing", False),
        # --packing auto could resolve on under flash attention, which is
        # incompatible with the sequential schedule order (explicit "on" was
        # already rejected by the schedule-mode validation).
        packing="off" if is_schedule else getattr(args, "packing", "auto"),
        max_length=getattr(args, "max_length", 2048),
        max_steps=overrides.max_steps,
        eval_steps=overrides.eval_steps,
        save_steps=overrides.save_steps,
        logging_steps=overrides.logging_steps,
        trainer_eval=trainer_eval_enabled,
        attn_implementation=selected_attn,
        learning_rate=getattr(args, "learning_rate", None),
        sequential_dataset=is_schedule,
    )
    if is_schedule:
        assert getattr(sft_config, "train_sampling_strategy", None) == "sequential", (
            "Schedule mode requires SFTConfig.train_sampling_strategy='sequential'; "
            "the installed TRL dropped or ignored the kwarg."
        )
        assert sft_config.shuffle_dataset is False, (
            "Schedule mode requires SFTConfig.shuffle_dataset=False; "
            "the installed TRL dropped or ignored the kwarg."
        )
    try:
        resume_checkpoint = _resolve_resume_checkpoint(
            output_dir,
            args.resume_from_checkpoint,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EVAL_INFRA_FAILURE_EXIT_CODE
    if resume_checkpoint is not None:
        logger.info("Resuming training from checkpoint: %s", resume_checkpoint)
        resume_key_error = _resume_checkpoint_model_key_error(model, resume_checkpoint)
        if resume_key_error is not None:
            logger.error(resume_key_error)
            return EVAL_INFRA_FAILURE_EXIT_CODE

    # --- Train ---
    logger.info("Starting training...")
    from trl import SFTTrainer

    _register_trl_flash_attention_variant(selected_attn)
    mixing_callback = None
    trainer_callbacks = None
    if is_schedule:
        from chess_llm.training.schedule_callback import MixingScheduleCallback

        mixing_callback = MixingScheduleCallback(
            schedule_plans,
            schedule_boundaries,
            # Forcing should_evaluate without an eval dataset would crash the
            # Trainer mid-run under --skip-trainer-eval.
            evaluate_at_boundaries=trainer_eval_enabled,
        )
        trainer_callbacks = [mixing_callback]

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=trainer_eval_ds,
        processing_class=tokenizer,
        callbacks=trainer_callbacks,
    )
    if mixing_callback is not None:
        _promote_callback_to_front(trainer, mixing_callback)

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_metrics = _augment_train_metrics_with_token_throughput(
        trainer,
        getattr(train_result, "metrics", {}) or {},
    )
    if train_metrics:
        logger.info("Training metrics: %s", train_metrics)
        if hasattr(trainer, "log_metrics"):
            trainer.log_metrics("train", train_metrics)
        if hasattr(trainer, "save_metrics"):
            trainer.save_metrics("train", train_metrics)

    # --- Save best model ---
    best_dir = output_dir / "best"
    logger.info("Saving best model to %s", best_dir)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # --- Post-training eval ---
    if not args.skip_eval:
        pred_path = output_dir / "eval_predictions.jsonl"
        baseline_path = _find_best_historical_baseline(args.output_root, phase.name)

        logger.info("Releasing training resources before benchmark evaluation")
        trainer, model = _release_training_resources(trainer, model)
        _clear_cuda_cache()

        eval_rc = _run_eval(
            str(best_dir), args.benchmark_dir, pred_path,
            baseline_path=baseline_path,
            phase=eval_phase,
            pass_k=8 if phase.name == "c" else 1,
            stockfish_path=args.stockfish_path,
            inference_backend=args.inference_backend,
            attn_implementation=args.attn_implementation,
            eval_batch_size=args.eval_batch_size,
            eval_max_new_tokens=args.eval_max_new_tokens,
            eval_acpl_depth=args.eval_acpl_depth,
            no_acpl=args.no_acpl,
            full_acpl_report=args.full_acpl_report,
            run_ledger=getattr(args, "run_ledger", None),
            artifact_mirror_dir=getattr(args, "artifact_mirror_dir", None),
            soft_gate=eval_soft_gate,
            max_examples_per_split=overrides.max_benchmark_examples_per_split,
            full_benchmark=eval_full_benchmark,
            wandb_project=args.wandb_project,
            wandb_run_name=(args.run_name or f"phase-{phase.name}-train") + "-post-eval",
            wandb_group=args.wandb_group,
            no_wandb=args.no_wandb,
            allow_wandb_offline=allow_wandb_offline,
        )
        failure_rc = _post_training_eval_failure_return_code(
            eval_rc,
            require_phase_gate=args.require_phase_gate,
        )
        if failure_rc is not None:
            if eval_rc >= EVAL_INFRA_FAILURE_EXIT_CODE:
                logger.error(
                    "Phase %s benchmark evaluation failed before metrics could be "
                    "trusted (exit code %d). Not marking checkpoint ready.",
                    phase.name,
                    eval_rc,
                )
            else:
                logger.error(
                    "Phase %s evaluation FAILED (exit code %d). "
                    "Review the report above before proceeding to the next phase.",
                    phase.name, eval_rc,
                )
            return failure_rc
        if eval_rc != 0:
            logger.warning(
                "Phase %s benchmark eval exited with %d. Keeping checkpoint "
                "ready because hard gates are disabled.",
                phase.name, eval_rc,
            )
        else:
            _warn_if_soft_gate_failures(
                pred_path,
                require_phase_gate=args.require_phase_gate,
            )

        # Write PASSED sentinel — required by resolve_checkpoint for next phase
        from chess_llm.training.phases import PHASE_PASSED_SENTINEL, PHASE_READY_SENTINEL
        ready_sentinel = output_dir / PHASE_READY_SENTINEL
        ready_sentinel.write_text(
            f"Phase {phase.name} checkpoint ready for continued training.\n",
        )
        logger.info("Wrote phase-ready sentinel: %s", ready_sentinel)
        if args.require_phase_gate:
            sentinel = output_dir / PHASE_PASSED_SENTINEL
            sentinel.write_text(f"Phase {phase.name} passed all checks.\n")
            logger.info("Wrote phase gate sentinel: %s", sentinel)
    else:
        from chess_llm.training.phases import PHASE_READY_SENTINEL
        ready_sentinel = output_dir / PHASE_READY_SENTINEL
        ready_sentinel.write_text(
            f"Phase {phase.name} checkpoint ready; benchmark eval skipped.\n",
        )
        logger.info("Wrote phase-ready sentinel: %s", ready_sentinel)
        if args.require_phase_gate:
            logger.warning(
                "Phase %s: --skip-eval used with --require-phase-gate. "
                "The next strict phase run will require a PASSED sentinel.",
                phase.name,
            )
        else:
            logger.warning(
                "Phase %s: --skip-eval used. The checkpoint is still ready "
                "for continued training, but no benchmark metrics were recorded.",
                phase.name,
            )

    logger.info("Phase %s complete.", phase.name)
    return 0


def _resolve_run_overrides(args: argparse.Namespace) -> RunOverrides:
    """Apply smoke-mode defaults and derive short-run trainer cadence."""
    max_train_examples = args.max_train_examples
    max_eval_examples = args.max_eval_examples
    max_benchmark_examples_per_split = args.max_benchmark_examples_per_split
    num_train_epochs = getattr(args, "num_train_epochs", None)
    max_steps = args.max_steps
    skip_trainer_eval = args.skip_trainer_eval

    if args.smoke_run:
        if max_train_examples is None:
            max_train_examples = SMOKE_MAX_TRAIN_EXAMPLES
        if max_eval_examples is None:
            max_eval_examples = SMOKE_MAX_EVAL_EXAMPLES
        if max_benchmark_examples_per_split is None:
            max_benchmark_examples_per_split = SMOKE_MAX_BENCHMARK_EXAMPLES_PER_SPLIT
        if max_steps is None:
            max_steps = SMOKE_MAX_STEPS
    elif max_eval_examples is None:
        max_eval_examples = DEFAULT_MAX_EVAL_EXAMPLES

    if max_eval_examples == 0:
        max_eval_examples = None

    eval_steps = args.trainer_eval_steps
    save_steps = args.trainer_save_steps
    logging_steps = None
    if max_steps is not None and 0 < max_steps <= SHORT_RUN_CADENCE_MAX_STEPS:
        short_run_interval = max(1, min(50, max_steps // 2))
        if eval_steps is None and not skip_trainer_eval:
            eval_steps = short_run_interval
        if save_steps is None and not skip_trainer_eval:
            save_steps = eval_steps
        logging_steps = _short_run_logging_steps(max_steps)
    elif eval_steps is None and save_steps is not None and not skip_trainer_eval:
        eval_steps = save_steps
    elif save_steps is None and not skip_trainer_eval:
        save_steps = eval_steps

    if (
        not skip_trainer_eval
        and eval_steps is not None
        and save_steps is not None
        and save_steps % eval_steps != 0
    ):
        raise ValueError(
            "--trainer-save-steps must be a multiple of --trainer-eval-steps "
            "when trainer eval is enabled.",
        )

    return RunOverrides(
        max_train_examples=max_train_examples,
        max_eval_examples=max_eval_examples,
        max_benchmark_examples_per_split=max_benchmark_examples_per_split,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        skip_trainer_eval=skip_trainer_eval,
    )


def _short_run_logging_steps(max_steps: int) -> int:
    """Keep bounded-run telemetry useful without logging every optimizer step."""
    return max(1, min(25, max_steps // 10))


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _format_epoch_count(value: float | int | None) -> str:
    if value is None:
        return "None"
    return f"{value:g}"


def _build_dry_run_training_details(
    phase,
    args: argparse.Namespace,
    overrides: RunOverrides,
    *,
    train_dataset_size: int,
) -> list[str]:
    from chess_llm.training.training_args import (
        DEFAULT_TRAINER_SAVE_STEPS,
        GRADIENT_ACCUMULATION_STEPS,
        PER_DEVICE_TRAIN_BATCH_SIZE,
        estimate_training_steps,
    )

    steps = estimate_training_steps(
        phase,
        train_dataset_size=train_dataset_size,
        num_train_epochs=overrides.num_train_epochs,
        max_steps=overrides.max_steps,
    )
    warmup_steps = 0
    if phase.warmup_ratio > 0 and steps > 0:
        import math

        warmup_steps = max(1, math.ceil(steps * phase.warmup_ratio))

    trainer_eval = "disabled" if overrides.skip_trainer_eval else "enabled"
    if overrides.skip_trainer_eval and overrides.save_steps is None:
        checkpoint_saves = "disabled during training"
    else:
        save_steps = overrides.save_steps or DEFAULT_TRAINER_SAVE_STEPS
        checkpoint_saves = f"every {save_steps} steps, keep 3 (resumable)"

    if args.no_wandb:
        wandb_line = "W&B: disabled"
    else:
        wandb_line = (
            f"W&B: enabled project={args.wandb_project} "
            f"group={args.wandb_group or '<none>'} "
            f"run={args.run_name or f'phase-{phase.name}-train'} "
            f"git={os.environ.get('WANDB_GIT_COMMIT') or '<unset>'}"
        )

    resume_arg = getattr(args, "resume_from_checkpoint", None)
    if resume_arg is None:
        resume_line = "Resume: disabled"
    elif resume_arg == "auto":
        resume_line = (
            "Resume: auto from latest checkpoint under "
            f"{args.output_root / f'phase_{phase.name}'}"
        )
    else:
        resume_line = f"Resume: {resume_arg}"

    return [
        f"Estimated optimizer steps: {steps}",
        f"Warmup steps: {warmup_steps} (ratio {phase.warmup_ratio:g})",
        f"Trainer eval: {trainer_eval}",
        f"Checkpoint saves: {checkpoint_saves}",
        wandb_line,
        f"Post-training benchmark eval: {'skipped' if args.skip_eval else 'enabled'}",
        resume_line,
        f"Requested attention implementation: {args.attn_implementation}",
        (
            "Training microbatch: "
            f"{PER_DEVICE_TRAIN_BATCH_SIZE} per device x "
            f"{GRADIENT_ACCUMULATION_STEPS} accumulation"
        ),
    ]


def _resolve_resume_checkpoint(output_dir: Path, requested: str | None) -> str | None:
    if requested is None:
        return None
    if requested == "auto":
        checkpoint = _find_latest_trainer_checkpoint(output_dir)
        if checkpoint is None:
            logger.warning(
                "Resume auto requested, but no checkpoint-N directory exists under %s; starting fresh",
                output_dir,
            )
            return None
        return str(checkpoint)

    checkpoint = Path(requested).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Resume checkpoint is not a directory: {checkpoint}")
    return str(checkpoint)


def _resume_checkpoint_model_key_error(model: Any, resume_checkpoint: str | None) -> str | None:
    """Return an error when resume checkpoint weights do not match the model."""
    if resume_checkpoint is None:
        return None
    checkpoint_keys = _checkpoint_model_keys(Path(resume_checkpoint))
    if checkpoint_keys is None:
        logger.warning(
            "Could not inspect model keys for resume checkpoint %s; "
            "falling back to Trainer resume loading.",
            resume_checkpoint,
        )
        return None

    model_keys = {str(key) for key in model.state_dict()}
    missing = sorted(model_keys.difference(checkpoint_keys))
    unexpected = sorted(checkpoint_keys.difference(model_keys))
    if not missing and not unexpected:
        return None

    parts = [
        "Resume checkpoint model keys do not match the loaded model; refusing "
        "to call Trainer.train(resume_from_checkpoint=...) because Transformers "
        "can otherwise downgrade this into a misleading fresh start.",
        f"{len(missing)} missing model key(s)",
        f"{len(unexpected)} unexpected checkpoint key(s)",
    ]
    if missing:
        parts.append(f"first missing model key: {missing[0]}")
    if unexpected:
        parts.append(f"first unexpected checkpoint key: {unexpected[0]}")
    return "; ".join(parts)


def _checkpoint_model_keys(checkpoint: Path) -> set[str] | None:
    """Read model weight names without loading tensor payloads when possible."""
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = checkpoint / index_name
        if not index_path.exists():
            continue
        with index_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        weight_map = payload.get("weight_map")
        if isinstance(weight_map, dict):
            return {str(key) for key in weight_map}

    safetensor_paths = sorted(checkpoint.glob("model*.safetensors"))
    if not safetensor_paths:
        return None

    try:
        from safetensors import safe_open
    except Exception:
        logger.warning("safetensors is unavailable; cannot inspect %s", checkpoint)
        return None

    keys: set[str] = set()
    for path in safetensor_paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys.update(str(key) for key in handle.keys())
    return keys


def _find_latest_trainer_checkpoint(output_dir: Path) -> Path | None:
    latest_step = -1
    latest_path: Path | None = None
    if not output_dir.exists():
        return None
    for candidate in output_dir.iterdir():
        if not candidate.is_dir() or not candidate.name.startswith("checkpoint-"):
            continue
        step_text = candidate.name.removeprefix("checkpoint-")
        if not step_text.isdigit():
            continue
        step = int(step_text)
        if step > latest_step:
            latest_step = step
            latest_path = candidate
    return latest_path


def _parse_task_upsample_overrides(values: list[str] | None) -> dict[str, int]:
    """Parse ``TASK=FACTOR`` CLI overrides for train-only task upsampling."""
    result: dict[str, int] = {}
    for value in values or []:
        task, sep, factor_text = value.partition("=")
        task = task.strip()
        if not sep or not task:
            raise ValueError(
                f"Invalid --task-upsample {value!r}; expected TASK=FACTOR",
            )
        try:
            factor = int(factor_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --task-upsample {value!r}; FACTOR must be an integer",
            ) from exc
        if factor < 1:
            raise ValueError(
                f"Invalid --task-upsample {value!r}; FACTOR must be >= 1",
            )
        if factor > 1:
            result[task] = factor
    return result


def _build_data_transform_config(args: argparse.Namespace):
    """Build row-level training-data transform controls from CLI args."""
    from chess_llm.training.data.loader import TrainingDataTransformConfig

    return TrainingDataTransformConfig(
        task_include=getattr(args, "task_include", []),
        task_exclude=getattr(args, "task_exclude", []),
        move_only_tasks=getattr(args, "move_only_task", []),
    )


def _data_transform_kwargs(data_transform_config) -> dict[str, Any]:
    """Return mixer kwargs only when data transforms are active."""
    if data_transform_config is None:
        return {}
    if getattr(data_transform_config, "is_default", False):
        return {}
    return {"data_transform_config": data_transform_config}


def _schedule_mode_config_error(args: argparse.Namespace) -> str | None:
    """Return a clear error for CLI flags that conflict with schedule mode."""
    if getattr(args, "require_phase_gate", False):
        return (
            "--require-phase-gate is not supported with --phase schedule: "
            "gates are measurements in schedule mode, not hard requirements."
        )
    if getattr(args, "packing", "auto") == "on":
        return (
            "--packing on is not supported with --phase schedule: packing "
            "reorders examples and would destroy the tier-mix schedule."
        )
    num_train_epochs = getattr(args, "num_train_epochs", None)
    if num_train_epochs is not None and float(num_train_epochs) != 1.0:
        return (
            "--num-train-epochs must be 1 (or omitted) with --phase schedule: "
            "the schedule is a single pass over the planned example budget."
        )
    return None


def _resolve_schedule_total_examples(
    schedule_total_examples: int | None,
    max_train_examples: int | None,
) -> int | None:
    """Cap the schedule example budget; None keeps the full-pool default."""
    values = [
        value
        for value in (schedule_total_examples, max_train_examples)
        if value is not None
    ]
    return min(values) if values else None


def _build_schedule_training_data(
    schedule,
    args: argparse.Namespace,
    overrides: RunOverrides,
    task_upsample: dict[str, int],
    output_dir: Path,
    *,
    data_transform_config=None,
):
    """Build the schedule train/eval datasets and persist the segment plan.

    The example budget (--schedule-total-examples capped by
    --max-train-examples) flows into ``build_schedule_dataset``; the train
    split is returned in schedule order and must not be limited or shuffled.
    """
    from chess_llm.training.data.mixer import build_schedule_dataset
    from chess_llm.training.schedule import boundary_steps

    total_examples = _resolve_schedule_total_examples(
        getattr(args, "schedule_total_examples", None),
        overrides.max_train_examples,
    )
    train_ds, eval_ds, plans = build_schedule_dataset(
        schedule,
        args.data_root,
        task_upsample=task_upsample,
        total_examples=total_examples,
        **_data_transform_kwargs(data_transform_config),
    )
    boundaries = boundary_steps(plans)
    _write_schedule_plan(
        output_dir,
        schedule,
        plans,
        boundaries,
        seed=42,
        total_examples=len(train_ds),
    )
    return train_ds, eval_ds, plans, boundaries


def _write_schedule_plan(
    output_dir: Path,
    schedule,
    plans,
    boundaries,
    *,
    seed: int,
    total_examples: int,
) -> Path:
    """Persist the concrete segment plan next to the run's checkpoints."""
    from chess_llm.training.schedule import DEFAULT_EFFECTIVE_BATCH

    payload = {
        "schedule": schedule.name,
        "seed": seed,
        "total_examples": total_examples,
        "effective_batch": DEFAULT_EFFECTIVE_BATCH,
        "boundary_steps": list(boundaries),
        "segments": [
            {
                "index": plan.index,
                "name": plan.name,
                "start_row": plan.start_row,
                "end_row": plan.end_row,
                "tier_weights": {
                    str(tier): weight for tier, weight in plan.tier_weights
                },
                "tier_rows": {str(tier): rows for tier, rows in plan.tier_rows},
            }
            for plan in plans
        ],
    }
    plan_path = output_dir / "schedule_plan.json"
    plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote schedule plan: %s", plan_path)
    return plan_path


def _schedule_dry_run(
    schedule,
    args: argparse.Namespace,
    overrides: RunOverrides,
    task_upsample: dict[str, int],
    data_transform_config,
) -> int:
    """Print the schedule data plan, boundaries, and warmup, then exit."""
    import math

    from chess_llm.training.data.mixer import summarize_schedule_data

    logger.info("--- DRY RUN: schedule data summary ---")
    total_examples = _resolve_schedule_total_examples(
        getattr(args, "schedule_total_examples", None),
        overrides.max_train_examples,
    )
    summary = summarize_schedule_data(
        schedule,
        args.data_root,
        task_upsample=task_upsample,
        total_examples=total_examples,
        **_data_transform_kwargs(data_transform_config),
    )
    logger.info("Total example budget: %d", summary["total_examples"])
    for key, count in sorted(summary["tier_pool_sizes"].items()):
        logger.info("  pool %-10s %10d", key, count)

    boundaries = summary["boundary_steps"]
    previous_boundary = 0
    for segment, boundary in zip(summary["segments"], boundaries):
        tier_bits = " ".join(
            f"{key}={count}" for key, count in sorted(segment["tier_rows"].items())
        )
        logger.info(
            "  segment %d %-12s rows %8d..%-8d steps %6d..%-6d %s",
            segment["index"],
            segment["name"],
            segment["start_row"],
            segment["end_row"],
            previous_boundary,
            boundary,
            tier_bits,
        )
        segment_steps = boundary - previous_boundary
        if segment_steps < 5000:
            logger.warning(
                "Segment %s spans only %d optimizer steps (< 5000); "
                "consider a larger example budget.",
                segment["name"],
                segment_steps,
            )
        previous_boundary = boundary

    total_steps = boundaries[-1] if boundaries else 0
    warmup_steps = (
        max(1, math.ceil(total_steps * schedule.warmup_ratio))
        if schedule.warmup_ratio > 0 and total_steps > 0
        else 0
    )
    logger.info("Estimated optimizer steps: %d (effective batch 32)", total_steps)
    logger.info("Warmup steps: %d (ratio %g)", warmup_steps, schedule.warmup_ratio)
    effective_learning_rate = (
        args.learning_rate if getattr(args, "learning_rate", None) is not None else schedule.learning_rate
    )
    logger.info("Learning rate: %s", effective_learning_rate)
    return 0


def _promote_callback_to_front(trainer, callback) -> bool:
    """Move ``callback`` ahead of the report_to integration callbacks.

    Trainer appends user callbacks after the W&B integration callback, so
    W&B would consume ``on_log`` before the schedule callback injects its
    ``schedule/*`` metrics. Repositioning to index 0 keeps the injected keys
    visible to W&B on the trainer's own step axis (chosen over having the
    callback call ``wandb.log(commit=False)`` directly).
    """
    handler = getattr(trainer, "callback_handler", None)
    callbacks = getattr(handler, "callbacks", None)
    if not isinstance(callbacks, list) or callback not in callbacks:
        logger.warning(
            "Could not reposition the schedule callback before integration "
            "callbacks; schedule/* metrics may be missing from W&B."
        )
        return False
    callbacks.remove(callback)
    callbacks.insert(0, callback)
    return True


def _limit_dataset(ds, max_examples: int | None, *, seed: int = 42):
    """Deterministically cap a dataset after mixing."""
    if max_examples is None or max_examples >= len(ds):
        return ds
    logger.info("Limiting dataset from %d to %d examples", len(ds), max_examples)
    return ds.shuffle(seed=seed).select(range(max_examples))


def _resolve_trainer_eval_dataset(eval_ds, *, skip_trainer_eval: bool):
    """Return the trainer eval dataset and whether trainer eval should be enabled."""
    if skip_trainer_eval or len(eval_ds) == 0:
        return None, False
    return eval_ds, True


def _configure_wandb_env(
    *,
    no_wandb: bool,
    wandb_project: str,
    wandb_group: str | None,
) -> None:
    """Make CLI W&B destination flags override stale shell environment."""
    if no_wandb:
        return
    os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_group:
        os.environ["WANDB_RUN_GROUP"] = wandb_group
        os.environ["WANDB_GROUP"] = wandb_group
    else:
        os.environ.pop("WANDB_RUN_GROUP", None)
        os.environ.pop("WANDB_GROUP", None)


def _release_training_resources(trainer, model) -> tuple[None, None]:
    """Drop trainer/model references that commonly retain GPU memory."""
    accelerator = getattr(trainer, "accelerator", None)
    free_memory = getattr(accelerator, "free_memory", None)
    if callable(free_memory):
        free_memory()
    try:
        setattr(trainer, "model", None)
    except Exception:
        pass
    return None, None


def _clear_cuda_cache(*, torch_module=None, collect=gc.collect) -> None:
    """Run Python GC and clear CUDA allocator caches when available."""
    collect()
    if torch_module is None:
        try:
            import torch as torch_module
        except Exception:
            return
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return
    is_available = getattr(cuda, "is_available", None)
    if callable(is_available) and not is_available():
        return
    empty_cache = getattr(cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()
    ipc_collect = getattr(cuda, "ipc_collect", None)
    if callable(ipc_collect):
        ipc_collect()


def _post_training_eval_failure_return_code(
    eval_rc: int,
    *,
    require_phase_gate: bool,
) -> int | None:
    """Return a fatal code for post-training eval, or None to keep training ready."""
    if eval_rc >= EVAL_INFRA_FAILURE_EXIT_CODE:
        return eval_rc
    if eval_rc != 0 and require_phase_gate:
        return eval_rc
    return None


def _warn_if_soft_gate_failures(
    pred_path: Path,
    *,
    require_phase_gate: bool,
) -> int | None:
    """Warn when a soft-gated eval returned success while recording failures."""
    if require_phase_gate:
        return None

    eval_run_path = pred_path.with_suffix(".eval_run.json")
    if not eval_run_path.exists():
        return None

    try:
        payload = json.loads(eval_run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read evaluation run metadata at %s: %s", eval_run_path, exc)
        return None

    try:
        n_failures = int(payload.get("n_failures", 0) or 0)
    except (TypeError, ValueError):
        return None
    if n_failures <= 0:
        return n_failures

    logger.warning(
        "Phase soft-gated benchmark eval recorded %d failure(s). Keeping "
        "checkpoint READY because hard gates are disabled. See %s",
        n_failures,
        eval_run_path,
    )
    return n_failures


def _find_best_historical_baseline(output_root: Path, current_phase: str) -> Path | None:
    """Build a best-historical baseline from all phases prior to ``current_phase``.

    Collects ``eval_predictions.results.json`` from phases that ran
    before ``current_phase`` and merges them by taking the per-metric
    maximum (minimum for ACPL and count metrics, which are lower-is-better).

    Always rebuilds from individual phase results to avoid including
    metrics from future phases (e.g. Phase C metrics leaking into a
    Phase A rerun baseline).

    Returns path to the merged baseline JSON, or None if no prior results.
    """
    import json

    phase_order = ["a", "b", "c"]
    if current_phase == "schedule":
        # Schedule runs sit outside the a->b->c progression: merge every
        # completed phase result as an informational best-ever reference.
        prior_phases = phase_order
    else:
        cutoff = phase_order.index(current_phase)
        prior_phases = phase_order[:cutoff]

    if not prior_phases:
        return None

    # Build from individual phase results (always, to stay phase-bounded)
    all_results: list[dict] = []
    for p in prior_phases:
        results_file = output_root / f"phase_{p}" / "eval_predictions.results.json"
        if results_file.exists():
            with open(results_file, encoding="utf-8") as fh:
                all_results.append(json.load(fh))

    if not all_results:
        return None

    # Merge: per split/metric, take the max value (best-ever)
    merged: dict[str, dict[str, float]] = {}
    for results in all_results:
        for split_name, metrics in results.items():
            if split_name not in merged:
                merged[split_name] = {}
            for metric_name, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                if "acpl" in metric_name or is_count_metric(metric_name):
                    # For ACPL and count metrics, lower is better — take the min
                    merged[split_name][metric_name] = min(
                        merged[split_name].get(metric_name, float("inf")), value,
                    )
                else:
                    merged[split_name][metric_name] = max(
                        merged[split_name].get(metric_name, 0.0), value,
                    )

    # Save phase-specific baseline (not a global best_baseline.json)
    baseline_path = output_root / f"baseline_for_phase_{current_phase}.json"
    with open(baseline_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    logger.info(
        "Built best-historical baseline from %d prior phase(s) [%s]: %s",
        len(all_results), ",".join(prior_phases), baseline_path,
    )
    return baseline_path


def _post_training_eval_preflight_error(
    benchmark_dir: Path,
    *,
    phase: str | None,
    full_benchmark: bool,
    inference_backend: str,
) -> str | None:
    """Return an infra error that would make post-training eval unusable."""
    backend_error = _eval_backend_preflight_error(inference_backend)
    if backend_error is not None:
        return backend_error

    benchmark_dir = Path(benchmark_dir)
    if not benchmark_dir.exists():
        return f"Benchmark directory does not exist: {benchmark_dir}"
    if not benchmark_dir.is_dir():
        return f"Benchmark path is not a directory: {benchmark_dir}"

    split_names = [path.stem for path in sorted(benchmark_dir.glob("*.jsonl"))]
    if not split_names:
        return f"Benchmark directory contains no JSONL split files: {benchmark_dir}"

    selected = _select_post_training_benchmark_split_names(
        split_names,
        phase=phase,
        full_benchmark=full_benchmark,
    )
    if not selected:
        return (
            "Benchmark directory contains no split files selected for "
            f"phase {phase or '<none>'}: {benchmark_dir}"
        )

    empty_splits = [
        split_name
        for split_name in selected
        if not _jsonl_has_nonblank_line(benchmark_dir / f"{split_name}.jsonl")
    ]
    if empty_splits:
        return (
            "Benchmark split file(s) are empty: "
            + ", ".join(f"{split_name}.jsonl" for split_name in empty_splits)
        )
    return None


def _eval_backend_preflight_error(inference_backend: str) -> str | None:
    if inference_backend != "vllm":
        return None
    if importlib.util.find_spec("vllm") is None:
        return "Eval backend 'vllm' was requested, but the vllm package is not importable."
    return None


def _select_post_training_benchmark_split_names(
    split_names: list[str],
    *,
    phase: str | None,
    full_benchmark: bool,
) -> list[str]:
    if phase and not full_benchmark:
        from chess_llm.training.evaluate import PHASE_DEFAULT_BENCHMARK_SPLITS

        requested = PHASE_DEFAULT_BENCHMARK_SPLITS.get(phase)
        if requested is not None:
            requested_set = set(requested)
            return [split_name for split_name in split_names if split_name in requested_set]
    return split_names


def _jsonl_has_nonblank_line(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            return any(line.strip() for line in handle)
    except OSError:
        return False


def _update_best_historical_baseline(output_root: Path, completed_phase: str) -> None:
    """No-op — baselines are now built on demand by _find_best_historical_baseline.

    Kept as a stub so callers don't need to change.  The per-phase
    results files in ``phase_{name}/eval_predictions.results.json`` are
    the source of truth; ``_find_best_historical_baseline`` merges them
    fresh each time to stay phase-bounded.
    """


def _run_eval(
    model_path: str,
    benchmark_dir: Path,
    pred_path: Path,
    *,
    baseline_path: Path | None = None,
    phase: str | None = None,
    pass_k: int = 1,
    stockfish_path: str | None = None,
    inference_backend: str = "transformers",
    attn_implementation: str = "auto",
    eval_batch_size: int = 16,
    eval_max_new_tokens: int = 512,
    eval_acpl_depth: int = 20,
    no_acpl: bool = False,
    full_acpl_report: bool = False,
    run_ledger: Path | None = None,
    artifact_mirror_dir: Path | None = None,
    soft_gate: bool = False,
    max_examples_per_split: int | None = None,
    full_benchmark: bool = False,
    wandb_project: str = "chess-sft",
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    no_wandb: bool = False,
    allow_wandb_offline: bool = False,
) -> int:
    """Run benchmark evaluation as a subprocess.

    Returns the subprocess exit code: 0 = all checks passed,
    non-zero = one or more criteria failed.
    """
    cmd = _build_eval_cmd(
        model_path,
        benchmark_dir,
        pred_path,
        baseline_path=baseline_path,
        phase=phase,
        pass_k=pass_k,
        stockfish_path=stockfish_path,
        inference_backend=inference_backend,
        attn_implementation=attn_implementation,
        eval_batch_size=eval_batch_size,
        eval_max_new_tokens=eval_max_new_tokens,
        eval_acpl_depth=eval_acpl_depth,
        no_acpl=no_acpl,
        full_acpl_report=full_acpl_report,
        run_ledger=run_ledger,
        artifact_mirror_dir=artifact_mirror_dir,
        soft_gate=soft_gate,
        max_examples_per_split=max_examples_per_split,
        full_benchmark=full_benchmark,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_group=wandb_group,
        no_wandb=no_wandb,
        allow_wandb_offline=allow_wandb_offline,
    )
    logger.info("Running evaluation: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


def _build_eval_cmd(
    model_path: str,
    benchmark_dir: Path,
    pred_path: Path,
    *,
    baseline_path: Path | None = None,
    phase: str | None = None,
    pass_k: int = 1,
    stockfish_path: str | None = None,
    inference_backend: str = "transformers",
    attn_implementation: str = "auto",
    eval_batch_size: int = 16,
    eval_max_new_tokens: int = 512,
    eval_acpl_depth: int = 20,
    no_acpl: bool = False,
    full_acpl_report: bool = False,
    run_ledger: Path | None = None,
    artifact_mirror_dir: Path | None = None,
    report_only: bool = False,
    soft_gate: bool = False,
    max_examples_per_split: int | None = None,
    full_benchmark: bool = False,
    wandb_project: str = "chess-sft",
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    no_wandb: bool = False,
    allow_wandb_offline: bool = False,
) -> list[str]:
    """Build the package evaluation subprocess command."""
    cmd = [
        sys.executable, "-m", "chess_llm.training.evaluate",
        "--model", str(model_path),
        "--benchmark-dir", str(benchmark_dir),
        "--output", str(pred_path),
        "--inference-backend", inference_backend,
        "--attn-implementation", attn_implementation,
        "--batch-size", str(eval_batch_size),
        "--max-new-tokens", str(eval_max_new_tokens),
        "--acpl-depth", str(eval_acpl_depth),
    ]
    if baseline_path is not None:
        cmd.extend(["--baseline", str(baseline_path)])
    if phase is not None:
        cmd.extend(["--phase", phase])
    if pass_k > 1:
        cmd.extend(["--pass-k", str(pass_k)])
    if stockfish_path is not None:
        cmd.extend(["--stockfish-path", stockfish_path])
    if no_acpl:
        cmd.append("--no-acpl")
    if full_acpl_report:
        cmd.append("--full-acpl-report")
    if run_ledger is not None:
        cmd.extend(["--run-ledger", str(run_ledger)])
    if artifact_mirror_dir is not None:
        cmd.extend(["--artifact-mirror-dir", str(artifact_mirror_dir)])
    if report_only:
        cmd.append("--report-only")
    if soft_gate:
        cmd.append("--soft-gate")
    if max_examples_per_split is not None:
        cmd.extend(["--max-examples-per-split", str(max_examples_per_split)])
    if full_benchmark:
        cmd.append("--full-benchmark")
    if no_wandb:
        cmd.append("--no-wandb")
    else:
        cmd.extend(["--wandb-project", wandb_project])
        if allow_wandb_offline:
            cmd.append("--allow-wandb-offline")
        if wandb_run_name is not None:
            cmd.extend(["--wandb-run-name", wandb_run_name])
        if wandb_group is not None:
            cmd.extend(["--wandb-group", wandb_group])
    return cmd


if __name__ == "__main__":
    sys.exit(main())
