"""Weights & Biases logging helpers for benchmark evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_wandb_payload(
    version: str,
    split_results: dict[str, dict[str, float]],
    split_counts: dict[str, int],
    *,
    n_failures: int,
    has_acpl: bool,
) -> dict[str, float | int | bool | str]:
    """Flatten benchmark results into a single W&B log payload."""
    payload: dict[str, float | int | bool | str] = {
        "benchmark/version": version,
        "phase_gate/failures": n_failures,
        "phase_gate/passed": n_failures == 0,
        "benchmark/has_acpl": has_acpl,
    }
    for split_name, count in split_counts.items():
        payload[f"benchmark/{split_name}/count"] = count
    for split_name, metrics in split_results.items():
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                payload[f"benchmark/{split_name}/{metric_name}"] = float(value)
    return payload


def wandb_preflight_error_code(message: str) -> str:
    """Classify W&B preflight failures for evaluation-run metadata."""
    if "WANDB_MODE=offline" in message or "WANDB_MODE=dryrun" in message:
        return "wandb_offline_disallowed"
    if "WANDB_MODE/WANDB_DISABLED disables it" in message:
        return "wandb_disabled"
    return "wandb_auth_missing"


def maybe_log_to_wandb(
    args: Any,
    *,
    version: str,
    split_results: dict[str, dict[str, float]],
    split_counts: dict[str, int],
    has_acpl: bool,
    n_failures: int,
    results_path: Path,
    eval_run_path: Path | None = None,
    prediction_analysis_path: Path | None = None,
) -> None:
    """Log eval metrics and files to W&B when enabled."""
    if args.no_wandb:
        return

    try:
        import wandb
    except ImportError:
        logger.warning("wandb is not installed; skipping eval logging")
        return

    payload = build_wandb_payload(
        version,
        split_results,
        split_counts,
        n_failures=n_failures,
        has_acpl=has_acpl,
    )
    config = {
        "model": args.model,
        "benchmark_dir": str(args.benchmark_dir),
        "phase": args.phase,
        "inference_backend": args.inference_backend,
        "attn_implementation": args.attn_implementation,
        "pass_k": args.pass_k,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "max_examples_per_split": args.max_examples_per_split,
        "stockfish_path": args.stockfish_path,
        "acpl_depth": args.acpl_depth,
        "acpl_enabled": not args.no_acpl,
        "full_acpl_report": args.full_acpl_report,
        "report_only": args.report_only,
        "soft_gate": args.soft_gate,
    }
    init_kwargs = {
        "project": args.wandb_project,
        "job_type": args.wandb_job_type,
        "config": config,
        "reinit": True,
    }
    if args.wandb_run_name:
        init_kwargs["name"] = args.wandb_run_name
    if args.wandb_group:
        init_kwargs["group"] = args.wandb_group

    run = wandb.init(**init_kwargs)
    if run is None:
        logger.warning("wandb.init() returned no run; skipping eval logging")
        return

    try:
        run.log(payload)
        for key, value in payload.items():
            run.summary[key] = value
        run.summary["artifacts/predictions_path"] = str(args.output)
        run.summary["artifacts/results_path"] = str(results_path)
        if eval_run_path is not None:
            run.summary["artifacts/eval_run_path"] = str(eval_run_path)
        if prediction_analysis_path is not None:
            run.summary["artifacts/prediction_analysis_path"] = str(prediction_analysis_path)

        artifact_name = args.wandb_run_name or f"benchmark-eval-{args.phase or 'adhoc'}"
        artifact = wandb.Artifact(artifact_name, type="benchmark-eval")
        artifact.add_file(str(args.output), name=args.output.name)
        artifact.add_file(str(results_path), name=results_path.name)
        if eval_run_path is not None:
            artifact.add_file(str(eval_run_path), name=eval_run_path.name)
        if prediction_analysis_path is not None:
            artifact.add_file(
                str(prediction_analysis_path),
                name=prediction_analysis_path.name,
            )
        run.log_artifact(artifact)
    finally:
        run.finish()

