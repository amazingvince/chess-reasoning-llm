"""Evaluation-run artifact construction and persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from chess_llm.artifacts.eval_runs import finalize_evaluation_artifacts
from chess_llm.artifacts.schemas import EvaluationRunArtifact
from chess_llm.training.eval_config import EvaluationConfig, EvaluationResult


def evaluation_result(
    config: EvaluationConfig,
    *,
    return_code: int,
    version: str = "unknown",
    split_results: dict[str, dict[str, float]] | None = None,
    split_counts: dict[str, int] | None = None,
    has_acpl: bool = False,
    n_failures: int = 0,
) -> EvaluationResult:
    return EvaluationResult(
        return_code=return_code,
        version=version,
        split_results={} if split_results is None else split_results,
        split_counts={} if split_counts is None else split_counts,
        has_acpl=has_acpl,
        n_failures=n_failures,
        predictions_path=config.output,
        results_path=config.output.with_suffix(".results.json"),
        eval_run_path=config.output.with_suffix(".eval_run.json"),
    )


def new_eval_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"eval-{timestamp}-{uuid4().hex[:8]}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluation_run_artifact(
    config: EvaluationConfig,
    result: EvaluationResult,
    *,
    primary_temperature: float,
    sample_temperature: float,
    metadata: dict[str, Any] | None = None,
) -> EvaluationRunArtifact:
    resolved_metadata = {
        "eval_run_path": str(result.eval_run_path),
        "wandb_project": config.wandb_project,
        "wandb_run_name": config.wandb_run_name,
        "wandb_group": config.wandb_group,
        "wandb_job_type": config.wandb_job_type,
        "wandb_enabled": not config.no_wandb,
        "wandb_offline_allowed": config.allow_wandb_offline,
        "require_benchmark_manifest": config.require_benchmark_manifest,
        "run_ledger_path": str(config.run_ledger) if config.run_ledger else None,
        "artifact_mirror_root": (
            str(config.artifact_mirror_dir) if config.artifact_mirror_dir else None
        ),
        "decision_rule": config.decision_rule,
    }
    if metadata:
        resolved_metadata.update(metadata)
    return EvaluationRunArtifact(
        run_id=new_eval_run_id(),
        created_at_utc=utc_now_iso(),
        model_id=config.model,
        phase=config.phase,
        benchmark_dir=str(config.benchmark_dir),
        benchmark_manifest_path=str(config.benchmark_dir / "manifest.json"),
        benchmark_version=result.version,
        predictions_path=str(result.predictions_path),
        results_path=str(result.results_path),
        return_code=result.return_code,
        split_counts=result.split_counts,
        has_acpl=result.has_acpl,
        n_failures=result.n_failures,
        inference={
            "backend": config.inference_backend,
            "attn_implementation": config.attn_implementation,
            "vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
            "vllm_max_model_len": config.vllm_max_model_len,
            "pass_k": config.pass_k,
            "primary_temperature": primary_temperature,
            "sample_temperature": sample_temperature,
            "max_new_tokens": config.max_new_tokens,
            "min_planning_max_new_tokens": config.min_planning_max_new_tokens,
            "batch_size": config.batch_size,
            "max_examples_per_split": config.max_examples_per_split,
            "splits": list(config.splits) if config.splits is not None else None,
            "effective_splits": sorted(result.split_counts),
            "full_benchmark": config.full_benchmark,
        },
        scoring={
            "stockfish_path": config.stockfish_path,
            "acpl_depth": config.acpl_depth,
            "acpl_workers": config.acpl_workers,
            "no_acpl": config.no_acpl,
            "full_acpl_report": config.full_acpl_report,
        },
        gate={
            "baseline_path": str(config.baseline) if config.baseline else None,
            "report_only": config.report_only,
            "soft_gate": config.soft_gate,
        },
        metadata=resolved_metadata,
    )


def write_evaluation_run_artifact(
    path: Path,
    artifact: EvaluationRunArtifact,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_evaluation_result_artifact(
    config: EvaluationConfig,
    result: EvaluationResult,
    *,
    primary_temperature: float = 0.0,
    sample_temperature: float = 0.7,
    metadata: dict[str, Any] | None = None,
) -> None:
    write_evaluation_run_artifact(
        result.eval_run_path,
        evaluation_run_artifact(
            config,
            result,
            primary_temperature=primary_temperature,
            sample_temperature=sample_temperature,
            metadata=metadata,
        ),
    )


def finalize_evaluation_result_artifacts(
    config: EvaluationConfig,
    result: EvaluationResult,
) -> None:
    finalize_evaluation_artifacts(
        result.eval_run_path,
        ledger_path=config.run_ledger,
        mirror_dir=config.artifact_mirror_dir,
    )

