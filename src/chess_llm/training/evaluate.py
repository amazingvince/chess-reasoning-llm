#!/usr/bin/env python3
"""Generate predictions on frozen benchmark and score them.

Usage:
    chess-llm-evaluate --model Qwen/Qwen3.5-0.8B \\
        --benchmark-dir chess_sft_data/benchmark \\
        --output baseline_predictions.jsonl

    chess-llm-evaluate --model chess_sft_checkpoints/phase_a/best \\
        --benchmark-dir chess_sft_data/benchmark \\
        --output phase_a_predictions.jsonl \\
        --pass-k 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from chess_llm.formats.prompts import SYSTEM_PROMPT
from chess_llm.core.legality import (
    parse_legality_reason_label,
    parse_legality_yes_no_answer,
)
from chess_llm.training.logging_utils import configure_cli_logging
from chess_llm.training.model_loading import (
    ATTENTION_IMPLEMENTATION_CHOICES,
    load_causal_lm_with_attention,
)
from chess_llm.training.vllm_export import prepare_model_for_vllm
from chess_llm.training.wandb_utils import wandb_config_error
from chess_llm.training.eval_exit_codes import (
    EVAL_INFRA_FAILURE_EXIT_CODE,
    EVAL_METRIC_FAILURE_EXIT_CODE,
    EVAL_SUCCESS_EXIT_CODE,
)

import chess
import chess.engine

from chess_llm.artifacts.schemas import EvaluationRunArtifact
from chess_llm.evals.benchmark import (
    BenchmarkExample,
    _UCI_RE,
    _extract_canonical_fen,
    _extract_fen_candidate,
    load_benchmark,
    score_split,
    score_prediction,
    centipawn_loss,
    extract_move,
    example_is_chess960,
    format_compliance,
    legal_move_rate,
    pass_at_k,
    normalize_prediction,
)
from chess_llm.evals.prediction_analysis import write_prediction_analysis_report

configure_cli_logging()
logger = logging.getLogger(__name__)


DEFAULT_STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish") or "stockfish"

# Task types that predict moves (candidates for ACPL) — matches run_benchmark.py
_MOVE_TASK_TYPES = frozenset({
    "best_move", "puzzle_solve", "endgame_best_move",
})

_ACPL_INVALID_MOVE_PENALTY = 150.0

PHASE_DEFAULT_BENCHMARK_SPLITS: dict[str, tuple[str, ...]] = {
    "a": ("perception", "rules"),
    "b": ("perception", "rules", "tactics", "evaluation", "openings", "endgames"),
    "c": (
        "perception",
        "rules",
        "tactics",
        "evaluation",
        "openings",
        "endgames",
        "planning",
    ),
}

PHASE_DEFAULT_BENCHMARK_TASK_TYPES: dict[str, dict[str, tuple[str, ...]]] = {
    "a": {
        "perception": (
            "board_print",
            "board_to_fen",
            "piece_id",
            "material_count",
            "material_inventory",
            "material_piece_counts",
            "material_value_totals",
            "material_balance_trace",
            "square_lookup",
            "rank_lookup",
            "square_coordinates",
            "fen_rank_expansion",
            "fen_rank_cell_edit",
            "fen_board_edit",
            "move_square_edits",
            "fen_assembly",
            "fen_row_application",
            "state_tracking",
        ),
        "rules": (
            "legal_moves",
            "side_piece_inventory",
            "piece_legal_moves",
            "piece_pseudo_legal_moves",
            "piece_legal_filter",
            "king_safety_filter",
            "legal_moves_by_piece",
            "check_detection",
            "special_rules",
            "legality_check",
        ),
    },
}


class _LazyAutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        from transformers import AutoTokenizer as _AutoTokenizer

        return _AutoTokenizer.from_pretrained(*args, **kwargs)


class _LazyAutoModelForCausalLM:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM

        return _AutoModelForCausalLM.from_pretrained(*args, **kwargs)


AutoTokenizer = _LazyAutoTokenizer
AutoModelForCausalLM = _LazyAutoModelForCausalLM


def _get_torch():
    import torch

    return torch


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
    max_new_tokens: int = 256
    batch_size: int = 16
    max_examples_per_split: int | None = None
    splits: tuple[str, ...] | None = None
    full_benchmark: bool = False
    stockfish_path: str = DEFAULT_STOCKFISH_PATH
    acpl_depth: int = 20
    no_acpl: bool = False
    full_acpl_report: bool = False
    baseline: Path | None = None
    phase: str | None = None
    report_only: bool = False
    soft_gate: bool = False
    wandb_project: str = "chess-sft"
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str = "benchmark-eval"
    no_wandb: bool = False
    allow_wandb_offline: bool = False

    def __post_init__(self) -> None:
        self.benchmark_dir = Path(self.benchmark_dir)
        self.output = Path(self.output)
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
            batch_size=args.batch_size,
            max_examples_per_split=args.max_examples_per_split,
            splits=tuple(args.splits) if args.splits else None,
            full_benchmark=args.full_benchmark,
            stockfish_path=args.stockfish_path,
            acpl_depth=args.acpl_depth,
            no_acpl=args.no_acpl,
            full_acpl_report=args.full_acpl_report,
            baseline=args.baseline,
            phase=args.phase,
            report_only=args.report_only,
            soft_gate=args.soft_gate,
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


def _evaluation_result(
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


def _new_eval_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"eval-{timestamp}-{uuid4().hex[:8]}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evaluation_run_artifact(
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
    }
    if metadata:
        resolved_metadata.update(metadata)
    return EvaluationRunArtifact(
        run_id=_new_eval_run_id(),
        created_at_utc=_utc_now_iso(),
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
            "batch_size": config.batch_size,
            "max_examples_per_split": config.max_examples_per_split,
            "splits": list(config.splits) if config.splits is not None else None,
            "effective_splits": sorted(result.split_counts),
            "full_benchmark": config.full_benchmark,
        },
        scoring={
            "stockfish_path": config.stockfish_path,
            "acpl_depth": config.acpl_depth,
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


def _write_evaluation_run_artifact(
    path: Path,
    artifact: EvaluationRunArtifact,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_evaluation_result_artifact(
    config: EvaluationConfig,
    result: EvaluationResult,
    *,
    primary_temperature: float = 0.0,
    sample_temperature: float = 0.7,
    metadata: dict[str, Any] | None = None,
) -> None:
    _write_evaluation_run_artifact(
        result.eval_run_path,
        _evaluation_run_artifact(
            config,
            result,
            primary_temperature=primary_temperature,
            sample_temperature=sample_temperature,
            metadata=metadata,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model on frozen benchmark")
    parser.add_argument("--model", required=True, help="HF model ID or local checkpoint path")
    parser.add_argument("--benchmark-dir", required=True, type=Path, help="Frozen benchmark directory")
    parser.add_argument("--output", required=True, type=Path, help="Output predictions JSONL path")
    parser.add_argument(
        "--inference-backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Generation backend to use for benchmark inference",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=ATTENTION_IMPLEMENTATION_CHOICES,
        default="auto",
        help="Attention backend for transformers model loads",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Target fraction of visible GPU memory vLLM may reserve (default: 0.85)",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help=(
            "Optional vLLM max_model_len override. Use this to avoid reserving "
            "the model's full context window for short benchmark prompts."
        ),
    )
    parser.add_argument("--pass-k", type=int, default=1, help="Number of samples per example for pass@k (default: 1)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (default: 0.0 for pass@1, 0.7 for pass@k)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate (default: 256)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for generation (default: 16)")
    parser.add_argument(
        "--max-examples-per-split", type=int, default=None,
        help="Limit benchmark examples loaded from each split (useful for smoke tests)",
    )
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        default=None,
        help=(
            "Benchmark split to evaluate. Repeat for multiple splits. "
            "Overrides phase-aware split defaults."
        ),
    )
    parser.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Evaluate every benchmark split instead of phase-aware default splits.",
    )
    parser.add_argument("--stockfish-path", type=str, default=DEFAULT_STOCKFISH_PATH, help="Path to Stockfish binary for ACPL (default: from STOCKFISH_PATH env or settings)")
    parser.add_argument("--acpl-depth", type=int, default=20, help="Stockfish search depth for ACPL (default: 20)")
    parser.add_argument("--no-acpl", action="store_true", help="Disable ACPL computation even if Stockfish is available")
    parser.add_argument(
        "--full-acpl-report",
        action="store_true",
        help="Compute ACPL on every split. Default is Phase C planning split only.",
    )
    parser.add_argument("--baseline", type=Path, default=None, help="Path to best-historical eval results JSON for regression checking")
    parser.add_argument("--phase", choices=["a", "b", "c"], default=None, help="Current training phase (enables phase-specific mandatory checks)")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Save/print metrics but skip phase-gate failure exit status",
    )
    parser.add_argument(
        "--soft-gate",
        action="store_true",
        help="Run phase criteria checks but return success even when they fail",
    )
    parser.add_argument("--wandb-project", type=str, default="chess-sft", help="W&B project name for eval logging")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name override for eval logging")
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional W&B group for eval logging")
    parser.add_argument("--wandb-job-type", type=str, default="benchmark-eval", help="W&B job type for eval logging")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging for this evaluation run")
    parser.add_argument(
        "--allow-wandb-offline",
        action="store_true",
        help="Allow WANDB_MODE=offline/dryrun for an intentional offline W&B eval run",
    )
    return parser.parse_args()


def load_prompt_tokenizer(model_path: str) -> AutoTokenizer:
    """Load the tokenizer used to format chat prompts."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise

        checkpoint_dir = Path(model_path)
        fallback_model = os.environ.get("CHESS_SFT_BASE_MODEL")
        if not checkpoint_dir.exists() or not checkpoint_dir.is_dir() or not fallback_model:
            raise

        logger.warning(
            "Tokenizer load failed for local checkpoint %s due to tokenizer config "
            "compatibility with the installed transformers version. Falling back to "
            "base tokenizer %s and reusing the checkpoint chat template.",
            model_path,
            fallback_model,
        )
        tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)

        chat_template_path = checkpoint_dir / "chat_template.jinja"
        if chat_template_path.exists():
            tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model_and_tokenizer(
    model_path: str,
    *,
    attn_implementation: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model with explicit attention selection and auto fallback."""
    torch = _get_torch()
    tokenizer = load_prompt_tokenizer(model_path)

    model_kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
    }

    # ``device_map='auto'`` is much slower than a plain CUDA load when the
    # process already sees exactly one GPU (the common Docker/WSL case here).
    # Keep ``device_map='auto'`` only when multiple GPUs are visible.
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        logger.info("Loading eval model directly onto CUDA")
        model, selected_attn = load_causal_lm_with_attention(
            AutoModelForCausalLM,
            model_path,
            model_kwargs,
            requested_attn=attn_implementation,
            logger=logger,
        )
        model = model.to("cuda")
    else:
        model_kwargs["device_map"] = "auto"
        model, selected_attn = load_causal_lm_with_attention(
            AutoModelForCausalLM,
            model_path,
            model_kwargs,
            requested_attn=attn_implementation,
            logger=logger,
        )
    logger.info("Selected attention implementation: %s", selected_attn or "<default>")
    model.eval()
    return model, tokenizer


_SYSTEM_PROMPT: str | None = None


def _get_system_prompt() -> str:
    """Return the package-owned system prompt used for benchmark chat prompts."""
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = SYSTEM_PROMPT
    return _SYSTEM_PROMPT


def format_prompt(example: BenchmarkExample, tokenizer: AutoTokenizer) -> str:
    """Format benchmark example as chat-templated prompt for generation."""
    user_content = example.prompt
    template_kwargs: dict[str, object] = {}

    # Qwen3 defaults to thinking mode and will happily burn the whole decode
    # budget on benchmark questions. Its chat template supports
    # ``enable_thinking=False`` directly; do not add a textual /no_think
    # directive because small SFT checkpoints can learn to echo it.
    tokenizer_name = (getattr(tokenizer, "name_or_path", "") or "").lower()
    if "qwen" in tokenizer_name:
        template_kwargs["enable_thinking"] = False

    messages = [
        {"role": "system", "content": _get_system_prompt()},
        {"role": "user", "content": user_content},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )


def _seed_generation(seed: int) -> None:
    """Seed CPU/GPU RNGs for reproducible sampled generation."""
    torch = _get_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_predictions_transformers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list[BenchmarkExample],
    *,
    num_samples: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Generate predictions for all examples with transformers.

    Returns mapping of example_id -> list of prediction strings.
    """
    from tqdm import tqdm

    predictions: dict[str, list[str]] = defaultdict(list)

    # Build prompts
    prompts = [format_prompt(ex, tokenizer) for ex in examples]

    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95

    torch = _get_torch()
    if num_samples > 1 and temperature == 0.0:
        for sample_idx in range(num_samples):
            logger.info("Deterministic sample %d/%d", sample_idx + 1, num_samples)
            single_predictions = generate_predictions_transformers(
                model,
                tokenizer,
                examples,
                num_samples=1,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                seed=seed + sample_idx,
            )
            for example_id, values in single_predictions.items():
                predictions[example_id].extend(values)
        return dict(predictions)

    return_sequences_per_prompt = 1
    effective_batch_size = batch_size
    if num_samples > 1 and temperature > 0.0:
        gen_kwargs["num_return_sequences"] = num_samples
        return_sequences_per_prompt = num_samples
        effective_batch_size = max(1, batch_size // num_samples)
        logger.info(
            "Generating %d sampled return sequences per prompt with effective batch size %d",
            num_samples,
            effective_batch_size,
        )
        _seed_generation(seed)

    for batch_start in tqdm(
        range(0, len(prompts), effective_batch_size),
        desc="Generating",
        total=(len(prompts) + effective_batch_size - 1) // effective_batch_size,
    ):
        batch_end = min(batch_start + effective_batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_examples = examples[batch_start:batch_end]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048 - max_new_tokens,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # Decode only the new tokens. Transformers returns outputs grouped by
        # prompt when num_return_sequences > 1.
        for i, ex in enumerate(batch_examples):
            input_len = inputs["input_ids"][i].shape[0]
            output_start = i * return_sequences_per_prompt
            output_end = output_start + return_sequences_per_prompt
            for output in outputs[output_start:output_end]:
                new_tokens = output[input_len:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                predictions[ex.example_id].append(text)

    return dict(predictions)


def _load_vllm_classes() -> tuple[type[Any], type[Any]]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM backend requested, but the 'vllm' package is not installed. "
            "Install it separately on a supported Linux/WSL GPU environment or "
            "use --inference-backend transformers."
        ) from exc
    return LLM, SamplingParams


class VllmPredictionGenerator:
    """Reusable vLLM offline generator for one evaluation run."""

    def __init__(
        self,
        model_path: str,
        tokenizer: AutoTokenizer,
        *,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        llm_cls: type[Any] | None = None,
        sampling_params_cls: type[Any] | None = None,
    ) -> None:
        using_default_llm_cls = llm_cls is None
        if llm_cls is None or sampling_params_cls is None:
            loaded_llm_cls, loaded_sampling_params_cls = _load_vllm_classes()
            llm_cls = llm_cls or loaded_llm_cls
            sampling_params_cls = sampling_params_cls or loaded_sampling_params_cls

        vllm_model_path = model_path
        if using_default_llm_cls:
            vllm_model_path = prepare_model_for_vllm(model_path)
            if vllm_model_path != model_path:
                logger.info("Prepared vLLM-compatible model export: %s", vllm_model_path)

        self.tokenizer = tokenizer
        self.sampling_params_cls = sampling_params_cls
        llm_kwargs: dict[str, Any] = {
            "model": vllm_model_path,
            "tokenizer": getattr(tokenizer, "name_or_path", model_path),
            "trust_remote_code": True,
            "generation_config": "vllm",
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        self.llm = llm_cls(**llm_kwargs)

    def generate(
        self,
        examples: list[BenchmarkExample],
        *,
        num_samples: int = 1,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        seed: int = 42,
    ) -> dict[str, list[str]]:
        prompts = [format_prompt(ex, self.tokenizer) for ex in examples]
        sampling_kwargs: dict = {
            "n": num_samples,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "seed": seed,
        }
        if temperature > 0.0:
            sampling_kwargs["top_p"] = 0.95

        outputs = self.llm.generate(prompts, self.sampling_params_cls(**sampling_kwargs))
        predictions: dict[str, list[str]] = {}
        for ex, output in zip(examples, outputs):
            predictions[ex.example_id] = [
                candidate.text for candidate in output.outputs
            ]
        return predictions


def generate_predictions_vllm(
    model_path: str,
    tokenizer: AutoTokenizer,
    examples: list[BenchmarkExample],
    *,
    num_samples: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    seed: int = 42,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int | None = None,
) -> dict[str, list[str]]:
    """Generate predictions with vLLM offline batched inference."""
    generator = VllmPredictionGenerator(
        model_path,
        tokenizer,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    return generator.generate(
        examples,
        num_samples=num_samples,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )


def save_predictions(
    predictions: dict[str, list[str]],
    path: Path,
    *,
    raw_predictions: dict[str, list[str]] | None = None,
    examples_by_id: Mapping[str, BenchmarkExample] | None = None,
) -> None:
    """Save predictions as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for example_id, preds in sorted(predictions.items()):
            example = examples_by_id.get(example_id) if examples_by_id else None
            raw_preds = raw_predictions.get(example_id, []) if raw_predictions else []
            for idx, pred in enumerate(preds):
                row = {
                    "example_id": example_id,
                    "sample_index": idx,
                    "prediction": pred,
                }
                if example is not None:
                    scoring_pred = raw_preds[idx] if idx < len(raw_preds) else pred
                    row.update(
                        {
                            "split": example.split,
                            "task_type": example.task_type,
                            "metric_type": example.metric_type,
                            "fen": example.fen,
                            "prompt": example.prompt,
                            "gold_answer": example.gold_answer,
                            "score": score_prediction(example, scoring_pred),
                        }
                    )
                    diagnostics = _prediction_diagnostics(example, scoring_pred)
                    if diagnostics:
                        row["diagnostics"] = diagnostics
                if idx < len(raw_preds) and raw_preds[idx] != pred:
                    row["raw_prediction"] = raw_preds[idx]
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Saved %d predictions to %s", sum(len(v) for v in predictions.values()), path)


def _prediction_diagnostics(
    example: BenchmarkExample,
    prediction: str,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    if example.metric_type == "fen_exact_match":
        fen_diagnostics = _fen_prediction_diagnostics(example, prediction)
        if fen_diagnostics:
            diagnostics["fen"] = fen_diagnostics
    if example.metric_type == "uci_set_jaccard":
        diagnostics["uci_set"] = _uci_set_prediction_diagnostics(example, prediction)
    if example.task_type == "legality_check":
        diagnostics["legality_check"] = _legality_check_prediction_diagnostics(
            example,
            prediction,
        )
    return diagnostics


def _fen_prediction_diagnostics(
    example: BenchmarkExample,
    prediction: str,
) -> dict[str, Any]:
    chess960 = example_is_chess960(example)
    pred_fen = _extract_canonical_fen(prediction, chess960=chess960)
    gold_fen = _extract_canonical_fen(example.gold_answer, chess960=chess960)
    pred_candidate = _extract_fen_candidate(prediction, chess960=chess960)
    gold_candidate = _extract_fen_candidate(example.gold_answer, chess960=chess960)
    pred_board_valid = _fen_board_is_valid(pred_candidate, chess960=chess960)
    gold_board_valid = _fen_board_is_valid(gold_candidate, chess960=chess960)
    result: dict[str, Any] = {
        "prediction_fen": pred_fen,
        "gold_fen": gold_fen,
        "prediction_fen_candidate": pred_candidate,
        "gold_fen_candidate": gold_candidate,
        "prediction_fen_syntax_valid": pred_candidate is not None,
        "gold_fen_syntax_valid": gold_candidate is not None,
        "prediction_fen_board_valid": pred_board_valid,
        "gold_fen_board_valid": gold_board_valid,
        "first4_match": False,
        "full_match": False,
    }
    if pred_fen is None or gold_fen is None:
        return result
    pred_fields = pred_fen.split()
    gold_fields = gold_fen.split()
    result["first4_match"] = pred_fields[:4] == gold_fields[:4]
    result["full_match"] = pred_fields == gold_fields
    return result


def _fen_board_is_valid(fen: str | None, *, chess960: bool = False) -> bool | None:
    if fen is None:
        return None
    try:
        return chess.Board(fen, chess960=chess960).is_valid()
    except (ValueError, TypeError):
        return False


def _uci_set_prediction_diagnostics(
    example: BenchmarkExample,
    prediction: str,
) -> dict[str, Any]:
    pred_set = set(_UCI_RE.findall(normalize_prediction(prediction).lower()))
    gold_set = set(_UCI_RE.findall(example.gold_answer.lower()))
    true_positive_count = len(pred_set & gold_set)
    extra_count = len(pred_set - gold_set)
    missing_count = len(gold_set - pred_set)

    diagnostics: dict[str, Any] = {
        "prediction_count": len(pred_set),
        "gold_count": len(gold_set),
        "true_positive_count": true_positive_count,
        "extra_count": extra_count,
        "missing_count": missing_count,
        "precision": true_positive_count / len(pred_set) if pred_set else 0.0,
        "recall": true_positive_count / len(gold_set) if gold_set else 0.0,
    }
    if example.task_type in {"legal_moves", "legal_moves_960"}:
        diagnostics["illegal_extra_count"] = _illegal_extra_uci_count(
            example,
            pred_set,
        )
    return diagnostics


def _legality_check_prediction_diagnostics(
    example: BenchmarkExample,
    prediction: str,
) -> dict[str, Any]:
    prediction_is_legal = parse_legality_yes_no_answer(normalize_prediction(prediction))
    gold_is_legal = parse_legality_yes_no_answer(example.gold_answer)
    prediction_reason = parse_legality_reason_label(prediction)
    gold_reason = (
        example.metadata.get("legality_reason_label")
        or parse_legality_reason_label(example.gold_answer)
    )
    return {
        "prediction_is_legal": prediction_is_legal,
        "gold_is_legal": gold_is_legal,
        "binary_match": (
            prediction_is_legal is not None
            and gold_is_legal is not None
            and prediction_is_legal == gold_is_legal
        ),
        "prediction_reason_label": prediction_reason,
        "gold_reason_label": gold_reason,
        "reason_match": (
            prediction_reason is not None
            and gold_reason is not None
            and prediction_reason == gold_reason
        ),
    }


def _illegal_extra_uci_count(
    example: BenchmarkExample,
    pred_set: set[str],
) -> int:
    try:
        board = chess.Board(example.fen, chess960=example_is_chess960(example))
    except (TypeError, ValueError):
        return 0
    legal_set = {move.uci() for move in board.legal_moves}
    return len(pred_set - legal_set)


def _merge_predictions_for_save(
    predictions: dict[str, list[str]],
    raw_predictions: dict[str, list[str]],
    sampled_predictions: dict[str, list[str]],
    raw_sampled_predictions: dict[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Merge greedy and sampled prediction lists without mutating inputs."""
    merged = {eid: list(preds) for eid, preds in predictions.items()}
    raw_merged = {eid: list(preds) for eid, preds in raw_predictions.items()}
    for eid, preds in sampled_predictions.items():
        merged.setdefault(eid, []).extend(preds)
    for eid, preds in raw_sampled_predictions.items():
        raw_merged.setdefault(eid, []).extend(preds)
    return merged, raw_merged


def _normalize_prediction_sets(
    predictions: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Apply benchmark output normalization to every generated prediction."""
    return {
        example_id: [normalize_prediction(pred) for pred in preds]
        for example_id, preds in predictions.items()
    }


def _score_split_with_protocol_predictions(
    examples: list[BenchmarkExample],
    normalized_predictions: dict[str, str],
    raw_predictions: dict[str, str],
    acpl_scores: dict[str, float] | None = None,
) -> dict[str, float]:
    """Score with raw outputs preserved for move-answer protocol metrics.

    ``score_prediction`` normalizes internally for primary answer metrics but
    intentionally checks ``<think>/<move>`` format and legal move extraction on
    the raw prediction.  Evaluation keeps normalized predictions for saving and
    answer scoring, so this helper restores raw text only for planning tasks
    that have protocol metrics.
    """
    scoring_predictions = dict(normalized_predictions)
    for example in examples:
        if example.task_type in ("best_move", "puzzle_solve"):
            raw = raw_predictions.get(example.example_id)
            if raw is not None:
                scoring_predictions[example.example_id] = raw
    return score_split(examples, scoring_predictions, acpl_scores)


def _open_stockfish(stockfish_path: str) -> chess.engine.SimpleEngine | None:
    """Try to open a Stockfish engine; return None on failure."""
    if not stockfish_path:
        logger.info("Stockfish path not configured; ACPL will be skipped")
        return None
    sf_path = Path(stockfish_path)
    resolved_path = str(sf_path) if sf_path.exists() else shutil.which(stockfish_path)
    if resolved_path is None:
        logger.info("Stockfish not found at %s; ACPL will be skipped", stockfish_path)
        return None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(resolved_path)
        logger.info("Stockfish loaded for ACPL from %s", resolved_path)
        return engine
    except Exception as exc:
        logger.warning("Could not load Stockfish: %s", exc)
        return None


def _evaluate_predicted_move(
    engine: chess.engine.SimpleEngine,
    fen: str,
    uci_move: str,
    depth: int = 20,
    chess960: bool = False,
) -> int | None:
    """Evaluate position after a predicted move using Stockfish.

    Returns centipawn score from the original side-to-move's perspective,
    or None if the move is invalid.
    """
    try:
        board = chess.Board(fen, chess960=chess960)
        original_turn = board.turn
        move = chess.Move.from_uci(uci_move)
        if uci_move not in {legal_move.uci() for legal_move in board.legal_moves}:
            return None
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].pov(original_turn)
        return score.score(mate_score=10000)
    except Exception:
        return None


def _evaluate_position(
    engine: chess.engine.SimpleEngine,
    fen: str,
    depth: int = 20,
    chess960: bool = False,
) -> int | None:
    """Evaluate a position using Stockfish.

    Returns centipawn score from the side-to-move's perspective,
    or None on error.  This IS the best-move eval because Stockfish's
    position score is the minimax value (eval achievable by best play).
    """
    try:
        board = chess.Board(fen, chess960=chess960)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].pov(board.turn)
        return score.score(mate_score=10000)
    except Exception:
        return None


def _example_is_chess960(example: BenchmarkExample) -> bool:
    return example_is_chess960(example)


def _legal_move_rate_for_example(
    example: BenchmarkExample,
    prediction: str,
) -> float | None:
    return legal_move_rate(
        prediction,
        example.fen,
        chess960=_example_is_chess960(example),
    )


def compute_acpl(
    engine: chess.engine.SimpleEngine,
    examples: list[BenchmarkExample],
    flat_preds: dict[str, str],
    depth: int = 20,
) -> dict[str, float]:
    """Compute per-example centipawn loss using Stockfish.

    ACPL = max(0, stockfish_best_eval - stockfish_eval_of_model_move)

    Both the position eval (best-move baseline) and the model-move eval
    are computed at the same depth by the same engine, eliminating
    depth-mismatch artifacts from pre-computed metadata.

    Returns mapping of example_id -> centipawn loss.
    """
    acpl_scores: dict[str, float] = {}
    best_cp_cache: dict[tuple[str, bool], int | None] = {}
    evaluated = 0
    missing_move = 0
    illegal_move = 0

    for ex in examples:
        if ex.task_type not in _MOVE_TASK_TYPES:
            continue
        chess960 = _example_is_chess960(ex)

        pred = flat_preds.get(ex.example_id, "")
        uci = extract_move(pred)
        if uci is None:
            # No valid move tag — apply penalty
            missing_move += 1
            acpl_scores[ex.example_id] = _ACPL_INVALID_MOVE_PENALTY
            continue

        try:
            board = chess.Board(ex.fen, chess960=chess960)
            move = chess.Move.from_uci(uci)
            if uci not in {legal_move.uci() for legal_move in board.legal_moves}:
                illegal_move += 1
                acpl_scores[ex.example_id] = _ACPL_INVALID_MOVE_PENALTY
                continue
        except Exception:
            illegal_move += 1
            acpl_scores[ex.example_id] = _ACPL_INVALID_MOVE_PENALTY
            continue

        best_cache_key = (ex.fen, chess960)
        if best_cache_key not in best_cp_cache:
            best_cp_cache[best_cache_key] = _evaluate_position(
                engine,
                ex.fen,
                depth,
                chess960=chess960,
            )
        best_cp = best_cp_cache[best_cache_key]
        if best_cp is None:
            continue

        predicted_cp = _evaluate_predicted_move(
            engine,
            ex.fen,
            uci,
            depth,
            chess960=chess960,
        )
        if predicted_cp is None:
            # Illegal move — apply penalty
            illegal_move += 1
            acpl_scores[ex.example_id] = _ACPL_INVALID_MOVE_PENALTY
            continue

        acpl_scores[ex.example_id] = centipawn_loss(float(best_cp), float(predicted_cp))
        evaluated += 1

    logger.info(
        "ACPL: evaluated %d positions with Stockfish (%d missing move, %d illegal move)",
        evaluated,
        missing_move,
        illegal_move,
    )
    return acpl_scores


def _select_acpl_splits(
    args: argparse.Namespace,
    split_examples: dict[str, list[BenchmarkExample]],
) -> set[str]:
    """Choose which splits should pay Stockfish ACPL cost."""
    if args.no_acpl:
        return set()
    if args.full_acpl_report:
        return set(split_examples)
    if args.phase == "c" and "planning" in split_examples:
        return {"planning"}
    return set()


def _resolve_benchmark_split_names(
    split_names: list[str],
    config: EvaluationConfig,
) -> list[str]:
    """Apply explicit or phase-aware split selection to discovered benchmark splits."""
    if config.splits is not None:
        requested = set(config.splits)
    elif config.phase and not config.full_benchmark:
        requested = set(PHASE_DEFAULT_BENCHMARK_SPLITS.get(config.phase, split_names))
    else:
        return split_names

    selected = [split_name for split_name in split_names if split_name in requested]
    missing = sorted(requested.difference(split_names))
    if missing:
        logger.warning(
            "Requested benchmark split(s) missing from %s: %s",
            config.benchmark_dir,
            ", ".join(missing),
        )
    logger.info(
        "Benchmark split selection: %s",
        ", ".join(selected) if selected else "<none>",
    )
    return selected


def _filter_phase_task_types(
    split_name: str,
    examples: list[BenchmarkExample],
    config: EvaluationConfig,
) -> list[BenchmarkExample]:
    """Drop task types outside the phase's default curriculum eval scope."""
    if config.phase is None or config.full_benchmark:
        return examples

    allowed_by_split = PHASE_DEFAULT_BENCHMARK_TASK_TYPES.get(config.phase, {})
    allowed = allowed_by_split.get(split_name)
    if allowed is None:
        return examples

    allowed_set = set(allowed)
    filtered = [example for example in examples if example.task_type in allowed_set]
    dropped = len(examples) - len(filtered)
    if dropped:
        logger.info(
            "Phase %s: filtered %d out-of-scope example(s) from split %s",
            config.phase.upper(),
            dropped,
            split_name,
        )
    return filtered


def _phase_allowed_task_types(
    split_name: str,
    config: EvaluationConfig,
) -> frozenset[str] | None:
    """Return phase-aware task types that should be loaded for a split."""
    if config.phase is None or config.full_benchmark:
        return None
    allowed_by_split = PHASE_DEFAULT_BENCHMARK_TASK_TYPES.get(config.phase, {})
    allowed = allowed_by_split.get(split_name)
    if allowed is None:
        return None
    return frozenset(allowed)


def _phase_gate_return_code(n_failures: int, *, soft_gate: bool) -> int:
    """Map phase-gate failures to a process return code."""
    if n_failures == 0 or soft_gate:
        return EVAL_SUCCESS_EXIT_CODE
    return EVAL_METRIC_FAILURE_EXIT_CODE


def print_report(
    version: str,
    split_results: dict[str, dict[str, float]],
    split_counts: dict[str, int],
    has_acpl: bool = False,
) -> None:
    """Print benchmark evaluation report."""
    print(f"\n{'=' * 60}")
    print(f"  Benchmark Evaluation ({version})")
    print(f"{'=' * 60}\n")

    split_order = [
        "perception", "rules", "tactics", "evaluation",
        "openings", "endgames", "planning", "chess960", "mate",
    ]

    for split_name in split_order:
        if split_name not in split_results:
            continue

        metrics = split_results[split_name]
        count = split_counts.get(split_name, 0)
        print(f"{split_name.capitalize()} ({count} examples):")

        for key, value in sorted(metrics.items()):
            if key in ("overall", "acpl"):
                continue
            if isinstance(value, float):
                if "acpl" in key:
                    print(f"  {key:<35} {value:>7.1f} cp")
                else:
                    print(f"  {key:<35} {value:>7.1%}")

        overall = metrics.get("overall")
        if overall is not None:
            print(f"  {'overall':<35} {overall:>7.1%}")

        acpl = metrics.get("acpl")
        if acpl is not None:
            print(f"  {'ACPL':<35} {acpl:>7.1f} cp")
        print()

    if not has_acpl:
        print(
            "Note: ACPL not computed "
            "(default: Phase C planning only; use --full-acpl-report for all splits)\n"
        )


def _build_wandb_payload(
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


def _wandb_preflight_error_code(message: str) -> str:
    """Classify W&B preflight failures for evaluation-run metadata."""
    if "WANDB_MODE=offline" in message or "WANDB_MODE=dryrun" in message:
        return "wandb_offline_disallowed"
    if "WANDB_MODE/WANDB_DISABLED disables it" in message:
        return "wandb_disabled"
    return "wandb_auth_missing"


def _maybe_log_to_wandb(
    args: argparse.Namespace,
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

    payload = _build_wandb_payload(
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


def run_evaluation(config: EvaluationConfig) -> EvaluationResult:
    """Run benchmark evaluation and return structured metrics plus exit policy."""
    args = config

    if not args.benchmark_dir.exists():
        print(f"[FAIL] Benchmark directory does not exist: {args.benchmark_dir}")
        result = _evaluation_result(
            args,
            return_code=EVAL_INFRA_FAILURE_EXIT_CODE,
        )
        _write_evaluation_result_artifact(
            args,
            result,
            metadata={"error": "benchmark_dir_missing"},
        )
        return result

    # Primary generation is always greedy (deterministic) for reliable gating.
    # pass@k uses a separate sampled generation pass (see below).
    primary_temperature = 0.0
    sample_temperature = args.temperature if args.temperature is not None else 0.7

    # Load manifest
    manifest_path = args.benchmark_dir / "manifest.json"
    manifest: dict = {}
    version = "unknown"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        version = manifest.get("version", "unknown")

    # Discover splits
    manifest_splits = manifest.get("splits", {})
    split_names = sorted(manifest_splits.keys()) if manifest_splits else [
        p.stem for p in sorted(args.benchmark_dir.glob("*.jsonl"))
    ]
    split_names = _resolve_benchmark_split_names(split_names, args)

    # Load all examples
    all_examples: list[BenchmarkExample] = []
    split_examples: dict[str, list[BenchmarkExample]] = {}
    for split_name in split_names:
        jsonl_path = args.benchmark_dir / f"{split_name}.jsonl"
        if not jsonl_path.exists():
            continue
        allowed_task_types = _phase_allowed_task_types(split_name, args)
        examples = load_benchmark(
            str(jsonl_path),
            max_examples=args.max_examples_per_split,
            task_types=allowed_task_types,
        )
        examples = _filter_phase_task_types(split_name, examples, args)
        if examples:
            split_examples[split_name] = examples
            all_examples.extend(examples)

    logger.info("Loaded %d examples across %d splits", len(all_examples), len(split_examples))
    if not all_examples:
        print(f"[FAIL] Benchmark directory contains no examples: {args.benchmark_dir}")
        result = _evaluation_result(
            args,
            return_code=EVAL_INFRA_FAILURE_EXIT_CODE,
            version=version,
        )
        _write_evaluation_result_artifact(
            args,
            result,
            metadata={"error": "benchmark_empty"},
        )
        return result

    wandb_error = wandb_config_error(
        no_wandb=args.no_wandb,
        allow_offline=args.allow_wandb_offline,
    )
    if wandb_error is not None:
        logger.error(wandb_error)
        print(f"[FAIL] {wandb_error}")
        result = _evaluation_result(
            args,
            return_code=EVAL_INFRA_FAILURE_EXIT_CODE,
            version=version,
        )
        _write_evaluation_result_artifact(
            args,
            result,
            metadata={
                "error": _wandb_preflight_error_code(wandb_error),
                "message": wandb_error,
            },
        )
        return result

    # Load inference backend only after validating that there is work to run.
    logger.info("Loading model: %s", args.model)
    logger.info("Inference backend: %s", args.inference_backend)
    vllm_generator: VllmPredictionGenerator | None = None
    if args.inference_backend == "transformers":
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            attn_implementation=args.attn_implementation,
        )
    else:
        model = None
        tokenizer = load_prompt_tokenizer(args.model)
        vllm_generator = VllmPredictionGenerator(
            args.model,
            tokenizer,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
        )

    # --- Primary pass: greedy generation for all examples ---
    # All gate metrics (ACPL, format, legal, regression) use this deterministic pass.
    if args.inference_backend == "transformers":
        raw_predictions = generate_predictions_transformers(
            model, tokenizer, all_examples,
            num_samples=1,
            temperature=primary_temperature,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            seed=42,
        )
    else:
        assert vllm_generator is not None
        raw_predictions = vllm_generator.generate(
            all_examples,
            num_samples=1,
            temperature=primary_temperature,
            max_new_tokens=args.max_new_tokens,
            seed=42,
        )
    predictions = _normalize_prediction_sets(raw_predictions)

    # --- Secondary pass: sampled generation for pass@k on puzzle examples only ---
    raw_sampled_predictions: dict[str, list[str]] = {}
    if args.pass_k > 1:
        puzzle_examples = [ex for ex in all_examples if ex.task_type == "puzzle_solve"]
        sampled_count = max(0, args.pass_k - 1)
        if puzzle_examples and sampled_count > 0:
            logger.info(
                "Generating %d sampled predictions for %d puzzle examples; "
                "greedy prediction is candidate 1 for pass@%d",
                sampled_count, len(puzzle_examples), args.pass_k,
            )
            if args.inference_backend == "transformers":
                raw_sampled_predictions = generate_predictions_transformers(
                    model, tokenizer, puzzle_examples,
                    num_samples=sampled_count,
                    temperature=sample_temperature,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                    seed=42,
                )
            else:
                assert vllm_generator is not None
                raw_sampled_predictions = vllm_generator.generate(
                    puzzle_examples,
                    num_samples=sampled_count,
                    temperature=sample_temperature,
                    max_new_tokens=args.max_new_tokens,
                    seed=42,
                )
    sampled_predictions = _normalize_prediction_sets(raw_sampled_predictions)

    # Save all predictions (greedy + sampled)
    all_preds_for_save, raw_all_preds_for_save = _merge_predictions_for_save(
        predictions,
        raw_predictions,
        sampled_predictions,
        raw_sampled_predictions,
    )
    save_predictions(
        all_preds_for_save,
        args.output,
        raw_predictions=raw_all_preds_for_save,
        examples_by_id={example.example_id: example for example in all_examples},
    )
    prediction_analysis_path = write_prediction_analysis_report(args.output)
    logger.info("Saved prediction analysis report to %s", prediction_analysis_path)

    # Flatten greedy predictions for scoring
    flat_preds: dict[str, str] = {}
    for eid, vals in predictions.items():
        flat_preds[eid] = vals[0]
    flat_raw_preds: dict[str, str] = {}
    for eid, vals in raw_predictions.items():
        flat_raw_preds[eid] = vals[0]

    # Open Stockfish only for splits that need ACPL. By default, that is the
    # Phase C planning gate, not every report split in every phase.
    engine: chess.engine.SimpleEngine | None = None
    has_acpl = False
    acpl_splits = _select_acpl_splits(args, split_examples)
    if acpl_splits:
        engine = _open_stockfish(args.stockfish_path)
        has_acpl = engine is not None
        if engine is not None:
            logger.info("ACPL enabled for splits: %s", ", ".join(sorted(acpl_splits)))
    else:
        logger.info(
            "ACPL disabled for this run. It is computed by default only for "
            "Phase C planning eval; use --full-acpl-report for all splits."
        )

    # Score each split
    split_results: dict[str, dict[str, float]] = {}
    split_counts: dict[str, int] = {}

    for split_name, examples in split_examples.items():
        # Compute ACPL if engine available
        acpl_scores: dict[str, float] | None = None
        if engine and split_name in acpl_splits:
            acpl_scores = compute_acpl(engine, examples, flat_preds, args.acpl_depth)

        metrics = _score_split_with_protocol_predictions(
            examples,
            flat_preds,
            flat_raw_preds,
            acpl_scores,
        )

        # pass@k for puzzle solving includes greedy first, then sampled candidates.
        puzzle_examples = [e for e in examples if e.task_type == "puzzle_solve"]
        if puzzle_examples and args.pass_k > 1:
            for k in (1, args.pass_k):
                hits = 0
                for ex in puzzle_examples:
                    preds = list(predictions.get(ex.example_id, [""]))
                    preds.extend(sampled_predictions.get(ex.example_id, []))
                    hits += pass_at_k(preds[:k], ex.gold_answer)
                metrics[f"puzzle_pass_at_{k}"] = hits / len(puzzle_examples)

        # Aggregate format/legal metrics for planning split
        planning_preds = [
            (ex, flat_raw_preds.get(ex.example_id, flat_preds.get(ex.example_id, "")))
            for ex in examples
            if ex.task_type in ("best_move", "puzzle_solve")
        ]
        if planning_preds:
            fc_scores = [format_compliance(p) for _, p in planning_preds]
            lm_scores = [
                v
                for v in (
                    legal_move_rate(
                        p,
                        ex.fen,
                        chess960=_example_is_chess960(ex),
                    )
                    for ex, p in planning_preds
                )
                if v is not None
            ]
            metrics["format_compliance"] = sum(fc_scores) / len(fc_scores)
            if lm_scores:
                metrics["legal_move_rate"] = sum(lm_scores) / len(lm_scores)

        split_results[split_name] = metrics
        split_counts[split_name] = len(examples)

    if engine:
        engine.quit()

    # Print report
    print_report(version, split_results, split_counts, has_acpl)

    # Load baseline for regression checking (if provided)
    baseline: dict[str, dict[str, float]] | None = None
    if args.baseline and args.baseline.exists():
        with open(args.baseline, encoding="utf-8") as fh:
            baseline = json.load(fh)
        logger.info("Loaded baseline results from %s", args.baseline)

    if args.report_only:
        logger.info("Report-only eval: skipping phase criteria failure status")
        n_failures = 0
    else:
        # Print phase criteria check (returns failure count)
        from chess_llm.training.phase_gate import check_phase_criteria
        n_failures = check_phase_criteria(
            split_results,
            baseline,
            phase=args.phase,
            has_acpl=has_acpl,
            allow_missing_criteria=(
                args.soft_gate and args.max_examples_per_split is not None
            ),
        )

    # Save results JSON for future regression checks
    results_path = args.output.with_suffix(".results.json")
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(split_results, fh, indent=2)
    logger.info("Saved results to %s (use as --baseline for next phase)", results_path)

    return_code = _phase_gate_return_code(n_failures, soft_gate=args.soft_gate)
    result = _evaluation_result(
        args,
        return_code=return_code,
        version=version,
        split_results=split_results,
        split_counts=split_counts,
        has_acpl=has_acpl,
        n_failures=n_failures,
    )
    _write_evaluation_result_artifact(
        args,
        result,
        primary_temperature=primary_temperature,
        sample_temperature=sample_temperature,
        metadata={"prediction_analysis_path": str(prediction_analysis_path)},
    )
    logger.info("Saved evaluation run metadata to %s", result.eval_run_path)

    _maybe_log_to_wandb(
        args,
        version=version,
        split_results=split_results,
        split_counts=split_counts,
        has_acpl=has_acpl,
        n_failures=n_failures,
        results_path=results_path,
        eval_run_path=result.eval_run_path,
        prediction_analysis_path=prediction_analysis_path,
    )

    return result


def _main_impl() -> int:
    args = parse_args()
    result = run_evaluation(EvaluationConfig.from_namespace(args))
    return result.return_code


def main() -> int:
    try:
        return _main_impl()
    except Exception as exc:
        logger.exception("Benchmark evaluation failed before metrics could be trusted")
        print(f"[FAIL] Benchmark evaluation infrastructure error: {exc}")
        return EVAL_INFRA_FAILURE_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
