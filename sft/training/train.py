#!/usr/bin/env python3
"""Main CLI for SFT training.

Usage:
    python train.py --phase a                        # Train Phase A
    python train.py --phase b                        # Train Phase B from A checkpoint
    python train.py --phase c                        # Train Phase C from B checkpoint
    python train.py --phase a --eval-only            # Baseline eval only (no training)
    python train.py --phase a --dry-run              # Print config + data summary, exit
    python train.py --phase a --smoke-run            # Short end-to-end smoke test
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Default paths (override with CLI args or env vars)
DEFAULT_DATA_ROOT = Path(os.environ.get("CHESS_SFT_OUTPUT", "E:/chess_sft_data")) / "output"
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("CHESS_SFT_CHECKPOINTS", "E:/chess_sft_checkpoints"))
DEFAULT_BENCHMARK_DIR = Path(os.environ.get("CHESS_SFT_OUTPUT", "E:/chess_sft_data")) / "benchmark"

DEFAULT_MAX_EVAL_EXAMPLES = 2048
SMOKE_MAX_TRAIN_EXAMPLES = 128
SMOKE_MAX_EVAL_EXAMPLES = 64
SMOKE_MAX_BENCHMARK_EXAMPLES_PER_SPLIT = 32
SMOKE_MAX_STEPS = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunOverrides:
    """Effective runtime limits after applying smoke-mode defaults."""

    max_train_examples: int | None = None
    max_eval_examples: int | None = None
    max_benchmark_examples_per_split: int | None = None
    max_steps: int | None = None
    eval_steps: int | None = None
    save_steps: int | None = None
    logging_steps: int | None = None
    skip_trainer_eval: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chess SFT training harness")
    parser.add_argument(
        "--phase", required=True, choices=["a", "b", "c"],
        help="Training phase (a=Foundation, b=Understanding, c=Planning)",
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
        "--max-steps", type=int, default=None,
        help="Override trainer max_steps; also tightens eval/save cadence for short runs",
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
    parser.add_argument("--wandb-project", type=str, default="chess-sft", help="W&B project name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B reporting (useful for CI)")
    parser.add_argument("--run-name", type=str, default=None, help="W&B run name override")
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional W&B group for training + eval runs")
    parser.add_argument(
        "--disable-liger-kernel",
        action="store_true",
        help="Disable Liger kernels for debugging or benchmarking",
    )
    parser.add_argument(
        "--inference-backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Generation backend for benchmark evaluation",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        default="auto",
        help="Attention backend for training and transformers eval model loads",
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
        default=256,
        help="Benchmark generation max_new_tokens (default: 256)",
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
        default=os.environ.get(
            "STOCKFISH_PATH",
            "C:/Users/amazi/Downloads/stockfish/stockfish/stockfish-windows-x86-64-avx512icl.exe",
        ),
        help="Path to Stockfish binary for ACPL (default: from STOCKFISH_PATH env)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides = _resolve_run_overrides(args)

    from config.phases import PHASES, resolve_checkpoint
    from data.mixer import build_phase_dataset, summarize_phase_data

    phase = PHASES[args.phase]
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
        logger.info("--- DRY RUN: data summary ---")
        summary = summarize_phase_data(phase, args.data_root)
        for key, count in sorted(summary.items()):
            logger.info("  %-12s %8d", key, count)
        logger.info("Learning rate: %s", phase.learning_rate)
        logger.info("Epochs: %d", phase.epochs)
        logger.info("Packing: False, max_length: 2048")
        logger.info("Effective batch size: 32 (4 * 8 accumulation)")
        logger.info("Eval backend: %s", args.inference_backend)
        if args.smoke_run or overrides.max_steps is not None:
            logger.info(
                "Runtime overrides: train<=%s eval<=%s benchmark/split<=%s max_steps=%s",
                overrides.max_train_examples,
                overrides.max_eval_examples,
                overrides.max_benchmark_examples_per_split,
                overrides.max_steps,
            )
        return 0

    # --- Eval only ---
    if args.eval_only:
        # Evaluate this phase's own best/ checkpoint if it exists,
        # otherwise fall back to the starting checkpoint (base model / prev phase).
        best_dir = output_dir / "best"
        eval_model = str(best_dir) if best_dir.exists() else model_path
        logger.info("Eval-only model: %s", eval_model)

        baseline_path = _find_best_historical_baseline(args.output_root, phase.name)
        return _run_eval(
            eval_model, args.benchmark_dir,
            output_dir / "eval_predictions.jsonl",
            baseline_path=baseline_path,
            phase=phase.name,
            pass_k=8 if phase.name == "c" else 1,
            stockfish_path=args.stockfish_path,
            inference_backend=args.inference_backend,
            attn_implementation=args.attn_implementation,
            eval_batch_size=args.eval_batch_size,
            eval_max_new_tokens=args.eval_max_new_tokens,
            eval_acpl_depth=args.eval_acpl_depth,
            no_acpl=args.no_acpl,
            full_acpl_report=args.full_acpl_report,
            soft_gate=not args.require_phase_gate,
            max_examples_per_split=overrides.max_benchmark_examples_per_split,
            wandb_project=args.wandb_project,
            wandb_run_name=args.run_name or f"phase-{phase.name}-eval-only",
            wandb_group=args.wandb_group,
            no_wandb=args.no_wandb,
        )

    # --- Build datasets ---
    logger.info("Building datasets...")
    train_ds, eval_ds = build_phase_dataset(phase, args.data_root)
    train_ds = _limit_dataset(train_ds, overrides.max_train_examples, seed=42)
    eval_ds = _limit_dataset(eval_ds, overrides.max_eval_examples, seed=42)
    if overrides.skip_trainer_eval:
        logger.info("Train: %d examples, trainer eval disabled", len(train_ds))
        trainer_eval_ds = None
    else:
        logger.info("Train: %d examples, Eval: %d examples", len(train_ds), len(eval_ds))
        trainer_eval_ds = eval_ds

    # --- Load model + tokenizer ---
    logger.info("Loading model and tokenizer from %s", model_path)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token: %s", tokenizer.pad_token)

    model_kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
    }
    from model_loading import load_causal_lm_with_attention

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
    from config.training_args import build_sft_config

    report_to: str | list[str] = "none" if args.no_wandb else "wandb"
    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_group:
            os.environ["WANDB_RUN_GROUP"] = args.wandb_group
            os.environ["WANDB_GROUP"] = args.wandb_group
    sft_config = build_sft_config(
        phase,
        output_dir,
        run_name=args.run_name,
        report_to=report_to,
        model_type=model_type,
        use_liger_kernel=not args.disable_liger_kernel,
        max_steps=overrides.max_steps,
        eval_steps=overrides.eval_steps,
        save_steps=overrides.save_steps,
        logging_steps=overrides.logging_steps,
        trainer_eval=not overrides.skip_trainer_eval,
    )

    # --- Train ---
    logger.info("Starting training...")
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=trainer_eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    # --- Save best model ---
    best_dir = output_dir / "best"
    logger.info("Saving best model to %s", best_dir)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # --- Post-training eval ---
    if not args.skip_eval:
        pred_path = output_dir / "eval_predictions.jsonl"
        baseline_path = _find_best_historical_baseline(args.output_root, phase.name)

        eval_rc = _run_eval(
            str(best_dir), args.benchmark_dir, pred_path,
            baseline_path=baseline_path,
            phase=phase.name,
            pass_k=8 if phase.name == "c" else 1,
            stockfish_path=args.stockfish_path,
            inference_backend=args.inference_backend,
            attn_implementation=args.attn_implementation,
            eval_batch_size=args.eval_batch_size,
            eval_max_new_tokens=args.eval_max_new_tokens,
            eval_acpl_depth=args.eval_acpl_depth,
            no_acpl=args.no_acpl,
            full_acpl_report=args.full_acpl_report,
            soft_gate=not args.require_phase_gate,
            max_examples_per_split=overrides.max_benchmark_examples_per_split,
            wandb_project=args.wandb_project,
            wandb_run_name=(args.run_name or f"phase-{phase.name}-train") + "-post-eval",
            wandb_group=args.wandb_group,
            no_wandb=args.no_wandb,
        )
        if eval_rc != 0 and args.require_phase_gate:
            logger.error(
                "Phase %s evaluation FAILED (exit code %d). "
                "Review the report above before proceeding to the next phase.",
                phase.name, eval_rc,
            )
            return eval_rc
        if eval_rc != 0:
            logger.warning(
                "Phase %s benchmark eval exited with %d. Keeping checkpoint "
                "ready because hard gates are disabled.",
                phase.name, eval_rc,
            )

        # Write PASSED sentinel — required by resolve_checkpoint for next phase
        from config.phases import PHASE_PASSED_SENTINEL, PHASE_READY_SENTINEL
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
        from config.phases import PHASE_READY_SENTINEL
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
    if max_steps is not None and max_steps > 0:
        short_run_interval = max(1, min(50, max_steps // 2))
        if eval_steps is None:
            eval_steps = short_run_interval
        if save_steps is None:
            save_steps = eval_steps
        logging_steps = 1
    elif eval_steps is None and save_steps is not None:
        eval_steps = save_steps
    elif save_steps is None:
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
        max_steps=max_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        skip_trainer_eval=skip_trainer_eval,
    )


def _limit_dataset(ds, max_examples: int | None, *, seed: int = 42):
    """Deterministically cap a dataset after mixing."""
    if max_examples is None or max_examples >= len(ds):
        return ds
    logger.info("Limiting dataset from %d to %d examples", len(ds), max_examples)
    return ds.shuffle(seed=seed).select(range(max_examples))


def _find_best_historical_baseline(output_root: Path, current_phase: str) -> Path | None:
    """Build a best-historical baseline from all phases prior to ``current_phase``.

    Collects ``eval_predictions.results.json`` from phases that ran
    before ``current_phase`` and merges them by taking the per-metric
    maximum (minimum for ACPL).

    Always rebuilds from individual phase results to avoid including
    metrics from future phases (e.g. Phase C metrics leaking into a
    Phase A rerun baseline).

    Returns path to the merged baseline JSON, or None if no prior results.
    """
    import json

    phase_order = ["a", "b", "c"]
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
                if "acpl" in metric_name:
                    # For ACPL, lower is better — take the min
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
    eval_max_new_tokens: int = 256,
    eval_acpl_depth: int = 20,
    no_acpl: bool = False,
    full_acpl_report: bool = False,
    soft_gate: bool = False,
    max_examples_per_split: int | None = None,
    wandb_project: str = "chess-sft",
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    no_wandb: bool = False,
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
        soft_gate=soft_gate,
        max_examples_per_split=max_examples_per_split,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_group=wandb_group,
        no_wandb=no_wandb,
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
    eval_max_new_tokens: int = 256,
    eval_acpl_depth: int = 20,
    no_acpl: bool = False,
    full_acpl_report: bool = False,
    report_only: bool = False,
    soft_gate: bool = False,
    max_examples_per_split: int | None = None,
    wandb_project: str = "chess-sft",
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    no_wandb: bool = False,
) -> list[str]:
    """Build the evaluate.py subprocess command."""
    evaluate_script = Path(__file__).parent / "evaluate.py"
    cmd = [
        sys.executable, str(evaluate_script),
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
    if report_only:
        cmd.append("--report-only")
    if soft_gate:
        cmd.append("--soft-gate")
    if max_examples_per_split is not None:
        cmd.extend(["--max-examples-per-split", str(max_examples_per_split)])
    if no_wandb:
        cmd.append("--no-wandb")
    else:
        cmd.extend(["--wandb-project", wandb_project])
        if wandb_run_name is not None:
            cmd.extend(["--wandb-run-name", wandb_run_name])
        if wandb_group is not None:
            cmd.extend(["--wandb-group", wandb_group])
    return cmd


if __name__ == "__main__":
    sys.exit(main())
