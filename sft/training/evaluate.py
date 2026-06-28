#!/usr/bin/env python3
"""Generate predictions on frozen benchmark and score them.

Usage:
    python evaluate.py --model Qwen/Qwen3-0.6B \\
        --benchmark-dir E:/chess_sft_data/benchmark \\
        --output baseline_predictions.jsonl

    python evaluate.py --model E:/chess_sft_checkpoints/phase_a/best \\
        --benchmark-dir E:/chess_sft_data/benchmark \\
        --output phase_a_predictions.jsonl \\
        --pass-k 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_loading import ATTENTION_IMPLEMENTATION_CHOICES, load_causal_lm_with_attention

# Add make_data to sys.path for benchmark imports.
# Prepend it so benchmark-time imports resolve make_data/config instead of
# sft/training/config when evaluate.py is launched from the training directory.
_MAKE_DATA_ROOT = Path(__file__).resolve().parent.parent / "make_data"
if str(_MAKE_DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAKE_DATA_ROOT))

import chess
import chess.engine

_SHADOWED_CONFIG_MODULES = {
    name: module
    for name, module in list(sys.modules.items())
    if name == "config" or name.startswith("config.")
}
for name in _SHADOWED_CONFIG_MODULES:
    sys.modules.pop(name, None)

try:
    from validation.benchmark import (
        BenchmarkExample,
        load_benchmark,
        score_split,
        centipawn_loss,
        format_compliance,
        legal_move_rate,
        pass_at_k,
        _extract_move,
        normalize_prediction,
    )
finally:
    for name in list(sys.modules):
        if name == "config" or name.startswith("config."):
            sys.modules.pop(name, None)
    sys.modules.update(_SHADOWED_CONFIG_MODULES)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_STOCKFISH_PATH = os.environ.get(
    "STOCKFISH_PATH",
    "/usr/games/stockfish" if os.name != "nt" else "C:/Users/amazi/Downloads/stockfish/stockfish/stockfish-windows-x86-64-avx512icl.exe",
)

# Task types that predict moves (candidates for ACPL) — matches run_benchmark.py
_MOVE_TASK_TYPES = frozenset({
    "best_move", "puzzle_solve", "endgame_best_move",
})

_ACPL_INVALID_MOVE_PENALTY = 150.0


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
    parser.add_argument("--pass-k", type=int, default=1, help="Number of samples per example for pass@k (default: 1)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (default: 0.0 for pass@1, 0.7 for pass@k)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate (default: 256)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for generation (default: 16)")
    parser.add_argument(
        "--max-examples-per-split", type=int, default=None,
        help="Limit benchmark examples loaded from each split (useful for smoke tests)",
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
    """Load system prompt from sft/make_data/config/system_prompt.py."""
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "make_data_system_prompt",
            _MAKE_DATA_ROOT / "config" / "system_prompt.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SYSTEM_PROMPT = mod.SYSTEM_PROMPT
    return _SYSTEM_PROMPT


def format_prompt(example: BenchmarkExample, tokenizer: AutoTokenizer) -> str:
    """Format benchmark example as chat-templated prompt for generation."""
    user_content = example.prompt
    template_kwargs: dict[str, object] = {}

    # Qwen3 defaults to thinking mode and will happily burn the whole decode
    # budget on benchmark questions. Force concise non-thinking mode for eval.
    tokenizer_name = (getattr(tokenizer, "name_or_path", "") or "").lower()
    if "qwen" in tokenizer_name:
        user_content = f"{user_content}\n/no_think"
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

    for sample_idx in range(num_samples):
        if num_samples > 1:
            logger.info("Sample %d/%d", sample_idx + 1, num_samples)
        if temperature > 0.0:
            _seed_generation(seed + sample_idx)

        for batch_start in tqdm(
            range(0, len(prompts), batch_size),
            desc=f"Generating (sample {sample_idx + 1})",
            total=(len(prompts) + batch_size - 1) // batch_size,
        ):
            batch_end = min(batch_start + batch_size, len(prompts))
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

            # Decode only the new tokens
            for i, (ex, output) in enumerate(zip(batch_examples, outputs)):
                input_len = inputs["input_ids"][i].shape[0]
                new_tokens = output[input_len:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                predictions[ex.example_id].append(text)

    return dict(predictions)


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
) -> dict[str, list[str]]:
    """Generate predictions with vLLM offline batched inference."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM backend requested, but the 'vllm' package is not installed. "
            "Install it separately on a supported Linux/WSL GPU environment or "
            "use --inference-backend transformers."
        ) from exc

    prompts = [format_prompt(ex, tokenizer) for ex in examples]
    llm = LLM(
        model=model_path,
        tokenizer=getattr(tokenizer, "name_or_path", model_path),
        trust_remote_code=True,
        generation_config="vllm",
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_kwargs: dict = {
        "n": num_samples,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "seed": seed,
    }
    if temperature > 0.0:
        sampling_kwargs["top_p"] = 0.95

    outputs = llm.generate(prompts, SamplingParams(**sampling_kwargs))
    predictions: dict[str, list[str]] = {}
    for ex, output in zip(examples, outputs):
        predictions[ex.example_id] = [candidate.text for candidate in output.outputs]
    return predictions


def save_predictions(
    predictions: dict[str, list[str]],
    path: Path,
    *,
    raw_predictions: dict[str, list[str]] | None = None,
) -> None:
    """Save predictions as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for example_id, preds in sorted(predictions.items()):
            raw_preds = raw_predictions.get(example_id, []) if raw_predictions else []
            for idx, pred in enumerate(preds):
                row = {
                    "example_id": example_id,
                    "prediction": pred,
                }
                if idx < len(raw_preds) and raw_preds[idx] != pred:
                    row["raw_prediction"] = raw_preds[idx]
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Saved %d predictions to %s", sum(len(v) for v in predictions.values()), path)


def _normalize_prediction_sets(
    predictions: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Apply benchmark output normalization to every generated prediction."""
    return {
        example_id: [normalize_prediction(pred) for pred in preds]
        for example_id, preds in predictions.items()
    }


def _open_stockfish(stockfish_path: str) -> chess.engine.SimpleEngine | None:
    """Try to open a Stockfish engine; return None on failure."""
    sf_path = Path(stockfish_path)
    if not sf_path.exists():
        logger.info("Stockfish not found at %s — ACPL will be skipped", sf_path)
        return None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
        logger.info("Stockfish loaded for ACPL from %s", sf_path)
        return engine
    except Exception as exc:
        logger.warning("Could not load Stockfish: %s", exc)
        return None


def _evaluate_predicted_move(
    engine: chess.engine.SimpleEngine,
    fen: str,
    uci_move: str,
    depth: int = 20,
) -> int | None:
    """Evaluate position after a predicted move using Stockfish.

    Returns centipawn score from the original side-to-move's perspective,
    or None if the move is invalid.
    """
    try:
        board = chess.Board(fen)
        original_turn = board.turn
        move = chess.Move.from_uci(uci_move)
        if move not in board.legal_moves:
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
) -> int | None:
    """Evaluate a position using Stockfish.

    Returns centipawn score from the side-to-move's perspective,
    or None on error.  This IS the best-move eval because Stockfish's
    position score is the minimax value (eval achievable by best play).
    """
    try:
        board = chess.Board(fen)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].pov(board.turn)
        return score.score(mate_score=10000)
    except Exception:
        return None


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
    best_cp_cache: dict[str, int | None] = {}
    evaluated = 0
    missing_move = 0
    illegal_move = 0

    for ex in examples:
        if ex.task_type not in _MOVE_TASK_TYPES:
            continue

        pred = flat_preds.get(ex.example_id, "")
        uci = _extract_move(pred)
        if uci is None:
            # No valid move tag — apply penalty
            missing_move += 1
            acpl_scores[ex.example_id] = _ACPL_INVALID_MOVE_PENALTY
            continue

        try:
            board = chess.Board(ex.fen)
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                illegal_move += 1
                acpl_scores[ex.example_id] = _ACPL_INVALID_MOVE_PENALTY
                continue
        except Exception:
            illegal_move += 1
            acpl_scores[ex.example_id] = _ACPL_INVALID_MOVE_PENALTY
            continue

        if ex.fen not in best_cp_cache:
            best_cp_cache[ex.fen] = _evaluate_position(engine, ex.fen, depth)
        best_cp = best_cp_cache[ex.fen]
        if best_cp is None:
            continue

        predicted_cp = _evaluate_predicted_move(engine, ex.fen, uci, depth)
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


def _phase_gate_return_code(n_failures: int, *, soft_gate: bool) -> int:
    """Map phase-gate failures to a process return code."""
    if soft_gate:
        return 0
    return 1 if n_failures > 0 else 0


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


def _maybe_log_to_wandb(
    args: argparse.Namespace,
    *,
    version: str,
    split_results: dict[str, dict[str, float]],
    split_counts: dict[str, int],
    has_acpl: bool,
    n_failures: int,
    results_path: Path,
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

        artifact_name = args.wandb_run_name or f"benchmark-eval-{args.phase or 'adhoc'}"
        artifact = wandb.Artifact(artifact_name, type="benchmark-eval")
        artifact.add_file(str(args.output), name=args.output.name)
        artifact.add_file(str(results_path), name=results_path.name)
        run.log_artifact(artifact)
    finally:
        run.finish()


def main() -> int:
    args = parse_args()

    if not args.benchmark_dir.exists():
        print(f"[FAIL] Benchmark directory does not exist: {args.benchmark_dir}")
        return 1

    # Primary generation is always greedy (deterministic) for reliable gating.
    # pass@k uses a separate sampled generation pass (see below).
    if args.temperature is not None:
        temperature = args.temperature
    else:
        temperature = 0.0

    # Load manifest
    manifest_path = args.benchmark_dir / "manifest.json"
    manifest: dict = {}
    version = "unknown"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        version = manifest.get("version", "unknown")

    # Load inference backend
    logger.info("Loading model: %s", args.model)
    logger.info("Inference backend: %s", args.inference_backend)
    if args.inference_backend == "transformers":
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            attn_implementation=args.attn_implementation,
        )
    else:
        model = None
        tokenizer = load_prompt_tokenizer(args.model)

    # Discover splits
    manifest_splits = manifest.get("splits", {})
    split_names = sorted(manifest_splits.keys()) if manifest_splits else [
        p.stem for p in sorted(args.benchmark_dir.glob("*.jsonl"))
    ]

    # Load all examples
    all_examples: list[BenchmarkExample] = []
    split_examples: dict[str, list[BenchmarkExample]] = {}
    for split_name in split_names:
        jsonl_path = args.benchmark_dir / f"{split_name}.jsonl"
        if not jsonl_path.exists():
            continue
        examples = load_benchmark(str(jsonl_path))
        if args.max_examples_per_split is not None and len(examples) > args.max_examples_per_split:
            logger.info(
                "Limiting split %s from %d to %d examples",
                split_name, len(examples), args.max_examples_per_split,
            )
            examples = examples[:args.max_examples_per_split]
        if examples:
            split_examples[split_name] = examples
            all_examples.extend(examples)

    logger.info("Loaded %d examples across %d splits", len(all_examples), len(split_examples))

    # --- Primary pass: greedy generation for all examples ---
    # All gate metrics (ACPL, format, legal, regression) use this deterministic pass.
    if args.inference_backend == "transformers":
        raw_predictions = generate_predictions_transformers(
            model, tokenizer, all_examples,
            num_samples=1,
            temperature=temperature,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            seed=42,
        )
    else:
        raw_predictions = generate_predictions_vllm(
            args.model, tokenizer, all_examples,
            num_samples=1,
            temperature=temperature,
            max_new_tokens=args.max_new_tokens,
            seed=42,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
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
                    temperature=0.7,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                    seed=42,
                )
            else:
                raw_sampled_predictions = generate_predictions_vllm(
                    args.model, tokenizer, puzzle_examples,
                    num_samples=sampled_count,
                    temperature=0.7,
                    max_new_tokens=args.max_new_tokens,
                    seed=42,
                    gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                )
    sampled_predictions = _normalize_prediction_sets(raw_sampled_predictions)

    # Save all predictions (greedy + sampled)
    all_preds_for_save = dict(predictions)
    raw_all_preds_for_save = dict(raw_predictions)
    for eid, preds in sampled_predictions.items():
        all_preds_for_save.setdefault(eid, []).extend(preds)
    for eid, preds in raw_sampled_predictions.items():
        raw_all_preds_for_save.setdefault(eid, []).extend(preds)
    save_predictions(
        all_preds_for_save,
        args.output,
        raw_predictions=raw_all_preds_for_save,
    )

    # Flatten greedy predictions for scoring
    flat_preds: dict[str, str] = {}
    for eid, vals in predictions.items():
        flat_preds[eid] = vals[0]

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

        metrics = score_split(examples, flat_preds, acpl_scores)

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
            (ex, flat_preds.get(ex.example_id, ""))
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
                        chess960=bool(ex.metadata.get("is_chess960")),
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
        from phase_gate import check_phase_criteria
        n_failures = check_phase_criteria(
            split_results, baseline, phase=args.phase, has_acpl=has_acpl,
        )

    # Save results JSON for future regression checks
    results_path = args.output.with_suffix(".results.json")
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(split_results, fh, indent=2)
    logger.info("Saved results to %s (use as --baseline for next phase)", results_path)

    _maybe_log_to_wandb(
        args,
        version=version,
        split_results=split_results,
        split_counts=split_counts,
        has_acpl=has_acpl,
        n_failures=n_failures,
        results_path=results_path,
    )

    return _phase_gate_return_code(n_failures, soft_gate=args.soft_gate)


if __name__ == "__main__":
    sys.exit(main())
