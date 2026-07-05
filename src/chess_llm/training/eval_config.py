"""Configuration and result types for benchmark evaluation."""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_STOCKFISH_PATH = (
    os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish") or "stockfish"
)


@dataclass
class EvaluationConfig:
    """Programmatic configuration for one benchmark evaluation run."""

    model: str
    benchmark_dir: Path
    output: Path
    inference_backend: str = "transformers"
    attn_implementation: str = "auto"
    vllm_gpu_memory_utilization: float = 0.85
    vllm_max_model_len: int | None = None
    pass_k: int = 1
    temperature: float | None = None
    max_new_tokens: int = 512
    min_planning_max_new_tokens: int = 256
    batch_size: int = 16
    max_examples_per_split: int | None = None
    splits: tuple[str, ...] | None = None
    full_benchmark: bool = False
    run_ledger: Path | None = None
    artifact_mirror_dir: Path | None = None
    decision_rule: str | None = None
    stockfish_path: str = DEFAULT_STOCKFISH_PATH
    acpl_depth: int = 20
    acpl_workers: int = 1
    no_acpl: bool = False
    full_acpl_report: bool = False
    baseline: Path | None = None
    phase: str | None = None
    report_only: bool = False
    soft_gate: bool = False
    require_benchmark_manifest: bool = False
    wandb_project: str = "chess-sft"
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str = "benchmark-eval"
    no_wandb: bool = False
    allow_wandb_offline: bool = False

    def __post_init__(self) -> None:
        self.benchmark_dir = Path(self.benchmark_dir)
        self.output = Path(self.output)
        if self.run_ledger is not None:
            self.run_ledger = Path(self.run_ledger)
        if self.artifact_mirror_dir is not None:
            self.artifact_mirror_dir = Path(self.artifact_mirror_dir)
        if self.baseline is not None:
            self.baseline = Path(self.baseline)

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "EvaluationConfig":
        """Build an evaluation config from CLI-parsed arguments."""
        return cls(
            model=args.model,
            benchmark_dir=args.benchmark_dir,
            output=args.output,
            inference_backend=args.inference_backend,
            attn_implementation=args.attn_implementation,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
            pass_k=args.pass_k,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            min_planning_max_new_tokens=getattr(
                args,
                "min_planning_max_new_tokens",
                256,
            ),
            batch_size=args.batch_size,
            max_examples_per_split=args.max_examples_per_split,
            splits=tuple(args.splits) if args.splits else None,
            full_benchmark=args.full_benchmark,
            run_ledger=getattr(args, "run_ledger", None),
            artifact_mirror_dir=getattr(args, "artifact_mirror_dir", None),
            decision_rule=getattr(args, "decision_rule", None),
            stockfish_path=args.stockfish_path,
            acpl_depth=args.acpl_depth,
            acpl_workers=getattr(args, "acpl_workers", 1),
            no_acpl=args.no_acpl,
            full_acpl_report=args.full_acpl_report,
            baseline=args.baseline,
            phase=args.phase,
            report_only=args.report_only,
            soft_gate=args.soft_gate,
            require_benchmark_manifest=getattr(
                args,
                "require_benchmark_manifest",
                False,
            ),
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_group=args.wandb_group,
            wandb_job_type=args.wandb_job_type,
            no_wandb=args.no_wandb,
            allow_wandb_offline=args.allow_wandb_offline,
        )


@dataclass(frozen=True)
class EvaluationResult:
    """Structured result from one benchmark evaluation run."""

    return_code: int
    version: str
    split_results: dict[str, dict[str, float]]
    split_counts: dict[str, int]
    has_acpl: bool
    n_failures: int
    predictions_path: Path
    results_path: Path
    eval_run_path: Path

