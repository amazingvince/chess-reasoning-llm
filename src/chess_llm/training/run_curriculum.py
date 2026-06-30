#!/usr/bin/env python3
"""Run the full SFT curriculum across phases."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from chess_llm.training.logging_utils import configure_cli_logging
from chess_llm.training.phases import PHASES, resolve_checkpoint
from chess_llm.training.train import (
    ATTENTION_IMPLEMENTATION_CHOICES,
    DEFAULT_BENCHMARK_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    _build_eval_cmd,
)

PHASE_ORDER = ["a", "b", "c"]
DEFAULT_PRE_EVAL_MAX_EXAMPLES_PER_SPLIT = 100

configure_cli_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full chess SFT curriculum")
    parser.add_argument("--start-phase", choices=PHASE_ORDER, default="a", help="First phase to run")
    parser.add_argument("--end-phase", choices=PHASE_ORDER, default="c", help="Last phase to run")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Tier data root")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Checkpoint/output root")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR, help="Frozen benchmark root")
    parser.add_argument("--smoke-run", action="store_true", help="Run bounded smoke settings for every phase")
    parser.add_argument(
        "--pre-eval",
        action="store_true",
        help="Run a lightweight report-only eval before each phase",
    )
    parser.add_argument(
        "--skip-pre-eval",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-train-examples", type=int, default=None, help="Limit train examples per phase")
    parser.add_argument(
        "--task-upsample",
        action="append",
        default=[],
        metavar="TASK=FACTOR",
        help=(
            "Upsample a specific task on each phase's train split before any "
            "max-train cap. Repeat for multiple tasks."
        ),
    )
    parser.add_argument("--max-eval-examples", type=int, default=None, help="Limit trainer eval examples per phase")
    parser.add_argument(
        "--max-benchmark-examples-per-split",
        type=int,
        default=None,
        help="Limit benchmark examples per split for both pre/post eval",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override trainer max_steps for every phase")
    parser.add_argument(
        "--trainer-eval-steps",
        type=int,
        default=None,
        help="Trainer loss-eval cadence in optimizer steps for every phase",
    )
    parser.add_argument(
        "--trainer-save-steps",
        type=int,
        default=None,
        help="Checkpoint save cadence in optimizer steps for every phase",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable activation checkpointing for every phase.",
    )
    parser.add_argument(
        "--skip-trainer-eval",
        action="store_true",
        help="Disable in-training loss eval for every phase",
    )
    parser.add_argument(
        "--require-phase-gate",
        action="store_true",
        help="Make benchmark failures fatal and require prior phases to have PASSED sentinels",
    )
    parser.add_argument(
        "--inference-backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Benchmark eval backend for all phases",
    )
    parser.add_argument("--stockfish-path", type=str, default=None, help="Stockfish binary path for ACPL")
    parser.add_argument(
        "--attn-implementation",
        choices=ATTENTION_IMPLEMENTATION_CHOICES,
        default="auto",
        help="Attention backend for training and transformers eval model loads",
    )
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Benchmark generation batch size")
    parser.add_argument("--eval-max-new-tokens", type=int, default=256, help="Benchmark generation max_new_tokens")
    parser.add_argument("--eval-acpl-depth", type=int, default=20, help="Benchmark Stockfish ACPL depth")
    parser.add_argument(
        "--enable-liger-fused-linear-ce",
        action="store_true",
        help="Enable Liger fused linear cross entropy for every phase.",
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
    parser.add_argument("--wandb-project", type=str, default="chess-sft", help="W&B project for train/eval runs")
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional W&B group for the whole curriculum")
    parser.add_argument("--run-prefix", type=str, default=None, help="Prefix for generated train/eval run names")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B for both training and eval")
    parser.add_argument(
        "--allow-wandb-offline",
        action="store_true",
        help=(
            "Allow WANDB_MODE=offline/dryrun for intentional offline W&B "
            "train/eval runs. Smoke runs allow this automatically."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned commands without executing them")
    return parser.parse_args()


def _phase_sequence(start_phase: str, end_phase: str) -> list[str]:
    start_idx = PHASE_ORDER.index(start_phase)
    end_idx = PHASE_ORDER.index(end_phase)
    if start_idx > end_idx:
        raise ValueError("--start-phase must be earlier than or equal to --end-phase")
    return PHASE_ORDER[start_idx:end_idx + 1]


def _curriculum_id(args: argparse.Namespace) -> str:
    if args.run_prefix:
        return args.run_prefix
    return f"curriculum-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _append_common_train_args(cmd: list[str], args: argparse.Namespace, *, run_name: str, wandb_group: str | None) -> list[str]:
    cmd.extend([
        "--data-root", str(args.data_root),
        "--output-root", str(args.output_root),
        "--benchmark-dir", str(args.benchmark_dir),
        "--inference-backend", args.inference_backend,
        "--attn-implementation", args.attn_implementation,
        "--eval-batch-size", str(args.eval_batch_size),
        "--eval-max-new-tokens", str(args.eval_max_new_tokens),
        "--eval-acpl-depth", str(args.eval_acpl_depth),
        "--run-name", run_name,
    ])
    if args.stockfish_path:
        cmd.extend(["--stockfish-path", args.stockfish_path])
    if args.smoke_run:
        cmd.append("--smoke-run")
    if args.max_train_examples is not None:
        cmd.extend(["--max-train-examples", str(args.max_train_examples)])
    for upsample_override in args.task_upsample:
        cmd.extend(["--task-upsample", upsample_override])
    if args.max_eval_examples is not None:
        cmd.extend(["--max-eval-examples", str(args.max_eval_examples)])
    if args.max_benchmark_examples_per_split is not None:
        cmd.extend(["--max-benchmark-examples-per-split", str(args.max_benchmark_examples_per_split)])
    if args.max_steps is not None:
        cmd.extend(["--max-steps", str(args.max_steps)])
    if args.trainer_eval_steps is not None:
        cmd.extend(["--trainer-eval-steps", str(args.trainer_eval_steps)])
    if args.trainer_save_steps is not None:
        cmd.extend(["--trainer-save-steps", str(args.trainer_save_steps)])
    if args.skip_trainer_eval:
        cmd.append("--skip-trainer-eval")
    if getattr(args, "gradient_checkpointing", False):
        cmd.append("--gradient-checkpointing")
    if getattr(args, "enable_liger_fused_linear_ce", False):
        cmd.append("--enable-liger-fused-linear-ce")
    if args.require_phase_gate:
        cmd.append("--require-phase-gate")
    if args.no_acpl:
        cmd.append("--no-acpl")
    if args.full_acpl_report:
        cmd.append("--full-acpl-report")
    if args.no_wandb:
        cmd.append("--no-wandb")
    else:
        cmd.extend(["--wandb-project", args.wandb_project])
        if args.allow_wandb_offline:
            cmd.append("--allow-wandb-offline")
        if wandb_group:
            cmd.extend(["--wandb-group", wandb_group])
    return cmd


def _build_train_phase_cmd(
    phase_name: str,
    args: argparse.Namespace,
    *,
    curriculum_id: str,
    wandb_group: str | None,
) -> list[str]:
    cmd = [sys.executable, "-m", "chess_llm.training.train", "--phase", phase_name]
    return _append_common_train_args(
        cmd,
        args,
        run_name=f"{curriculum_id}-phase-{phase_name}-train",
        wandb_group=wandb_group,
    )


def _build_pre_eval_cmd(
    phase_name: str,
    model_path: str,
    args: argparse.Namespace,
    *,
    curriculum_id: str,
    wandb_group: str | None,
) -> list[str]:
    output_dir = args.output_root / f"phase_{phase_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "pre_eval_predictions.jsonl"
    max_examples_per_split = (
        args.max_benchmark_examples_per_split
        if args.max_benchmark_examples_per_split is not None
        else DEFAULT_PRE_EVAL_MAX_EXAMPLES_PER_SPLIT
    )
    return _build_eval_cmd(
        model_path,
        args.benchmark_dir,
        pred_path,
        phase=phase_name,
        pass_k=1,
        stockfish_path=args.stockfish_path,
        inference_backend=args.inference_backend,
        attn_implementation=args.attn_implementation,
        eval_batch_size=args.eval_batch_size,
        eval_max_new_tokens=args.eval_max_new_tokens,
        eval_acpl_depth=args.eval_acpl_depth,
        no_acpl=True,
        report_only=True,
        max_examples_per_split=max_examples_per_split,
        wandb_project=args.wandb_project,
        wandb_run_name=f"{curriculum_id}-phase-{phase_name}-pre-eval",
        wandb_group=wandb_group,
        no_wandb=args.no_wandb,
        allow_wandb_offline=(
            args.allow_wandb_offline or getattr(args, "smoke_run", False)
        ),
    )


def _run_command(cmd: list[str], *, dry_run: bool) -> int:
    logger.info("Running: %s", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd).returncode


def _resolve_phase_start_model(
    phase_name: str,
    output_root: Path,
    *,
    dry_run: bool,
    planned_completed_phases: set[str],
    require_passed: bool = False,
) -> str:
    """Resolve the model to evaluate/train from for a phase.

    During dry runs, treat earlier phases in the same planned sequence as if
    they will finish successfully so we can print the full A -> B -> C plan.
    """
    phase = PHASES[phase_name]
    if not dry_run or phase_name == "a":
        return resolve_checkpoint(phase, output_root, require_passed=require_passed)

    prev_phase = PHASE_ORDER[PHASE_ORDER.index(phase_name) - 1]
    if prev_phase in planned_completed_phases:
        return str(output_root / f"phase_{prev_phase}" / "best")
    return resolve_checkpoint(phase, output_root, require_passed=require_passed)


def main() -> int:
    args = parse_args()
    phases = _phase_sequence(args.start_phase, args.end_phase)
    curriculum_id = _curriculum_id(args)
    wandb_group = None if args.no_wandb else (args.wandb_group or curriculum_id)

    logger.info("Curriculum phases: %s", ", ".join(phases))
    logger.info("Run prefix: %s", curriculum_id)
    if wandb_group:
        logger.info("W&B group: %s", wandb_group)

    planned_completed_phases: set[str] = set()
    for phase_name in phases:
        phase = PHASES[phase_name]
        logger.info("=" * 60)
        logger.info("Phase %s: %s", phase.name.upper(), phase.display_name)
        logger.info("=" * 60)

        start_model = _resolve_phase_start_model(
            phase_name,
            args.output_root,
            dry_run=args.dry_run,
            planned_completed_phases=planned_completed_phases,
            require_passed=args.require_phase_gate,
        )
        logger.info("Phase %s starting model: %s", phase.name.upper(), start_model)

        if args.pre_eval and not args.skip_pre_eval:
            pre_eval_cmd = _build_pre_eval_cmd(
                phase_name,
                start_model,
                args,
                curriculum_id=curriculum_id,
                wandb_group=wandb_group,
            )
            pre_eval_rc = _run_command(pre_eval_cmd, dry_run=args.dry_run)
            if pre_eval_rc != 0:
                logger.warning(
                    "Phase %s pre-eval returned %d. Continuing into training.",
                    phase.name.upper(),
                    pre_eval_rc,
                )

        train_cmd = _build_train_phase_cmd(
            phase_name,
            args,
            curriculum_id=curriculum_id,
            wandb_group=wandb_group,
        )
        train_rc = _run_command(train_cmd, dry_run=args.dry_run)
        if train_rc != 0:
            logger.error("Phase %s failed with exit code %d", phase.name.upper(), train_rc)
            return train_rc
        planned_completed_phases.add(phase_name)

    logger.info("Curriculum completed successfully: %s", ", ".join(phases))
    return 0


if __name__ == "__main__":
    sys.exit(main())
