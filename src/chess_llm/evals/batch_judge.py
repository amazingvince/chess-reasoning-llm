"""Batch export benchmark predictions into rollout and judgment artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from chess_llm.artifacts.jsonl import write_jsonl
from chess_llm.artifacts.schemas import JudgmentArtifact, PromptArtifact, RolloutArtifact
from chess_llm.autodata.judge import judge_rollout
from chess_llm.autodata.rollouts import build_rollout
from chess_llm.autodata.stockfish_judge import judge_rollout_with_stockfish
from chess_llm.evals.benchmark_artifacts import load_benchmark_prompts
from chess_llm.external.stockfish import (
    StockfishEngineConfig,
    configure_stockfish_engine,
    open_stockfish,
    stockfish_engine_name,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchJudgeResult:
    """Paths and counts produced by one batch judge run."""

    prompts_path: Path
    rollouts_path: Path
    judgments_path: Path
    manifest_path: Path
    prompt_count: int
    rollout_count: int
    judgment_count: int
    splits: list[str]


def judge_prediction_file(
    benchmark_dir: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    model_id: str,
    splits: Iterable[str] | None = None,
    *,
    stockfish_path: str | Path | None = None,
    stockfish_depth: int = 20,
    stockfish_threads: int = 1,
    stockfish_hash_mb: int = 256,
    syzygy_path: str | Path | None = None,
    chess960: bool = False,
    analysis_engine: Any | None = None,
) -> BatchJudgeResult:
    """Convert benchmark predictions into prompt, rollout, and judgment files."""
    split_list = list(splits) if splits is not None else None
    prompts_by_id = load_benchmark_prompts(benchmark_dir, split_list)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    prompts: list[PromptArtifact] = []
    seen_prompt_ids: set[str] = set()
    rollouts: list[RolloutArtifact] = []
    judgments: list[JudgmentArtifact] = []
    engine = analysis_engine
    engine_name: str | None = None
    opened_engine = False
    if engine is None and stockfish_path is not None:
        engine = open_stockfish(
            StockfishEngineConfig(
                path=stockfish_path,
                threads=stockfish_threads,
                hash_mb=stockfish_hash_mb,
                syzygy_path=syzygy_path,
            )
        )
        opened_engine = True
    elif engine is not None:
        configure_stockfish_engine(
            engine,
            threads=stockfish_threads,
            hash_mb=stockfish_hash_mb,
            syzygy_path=syzygy_path,
        )

    prediction_path = Path(predictions_path)
    skipped_out_of_split = 0
    try:
        with prediction_path.open(encoding="utf-8") as fh:
            for row_index, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                example_id = str(row.get("example_id", ""))
                if example_id not in prompts_by_id:
                    if split_list is not None:
                        skipped_out_of_split += 1
                        continue
                    raise ValueError(
                        f"prediction example_id {example_id!r} was not found in "
                        f"loaded benchmark prompts"
                    )

                prompt = prompts_by_id[example_id]
                if prompt.prompt_id not in seen_prompt_ids:
                    prompts.append(prompt)
                    seen_prompt_ids.add(prompt.prompt_id)

                raw_output = str(row.get("raw_prediction", row.get("prediction", "")))
                rollout_id = f"rollout-{row_index:06d}-{example_id}"
                rollout = build_rollout(
                    prompt,
                    model_id,
                    raw_output,
                    rollout_id=rollout_id,
                    metadata={
                        "example_id": example_id,
                        "prediction": row.get("prediction"),
                        "prediction_index": row_index,
                        "source_predictions_path": str(prediction_path),
                        "raw_prediction_present": "raw_prediction" in row,
                    },
                )
                rollouts.append(rollout)

                judgment_metadata = {
                    "example_id": example_id,
                    "prediction_index": row_index,
                    "source_predictions_path": str(prediction_path),
                }
                prompt_chess960 = _prompt_chess960(prompt, default=chess960)
                if engine is None:
                    judgment = judge_rollout(
                        prompt,
                        rollout,
                        judgment_id=f"judgment-{row_index:06d}-{example_id}",
                        chess960=prompt_chess960,
                        metadata=judgment_metadata,
                    )
                else:
                    judgment = judge_rollout_with_stockfish(
                        prompt,
                        rollout,
                        engine,
                        judgment_id=f"judgment-{row_index:06d}-{example_id}",
                        depth=stockfish_depth,
                        chess960=prompt_chess960,
                        metadata=judgment_metadata,
                    )
                judgments.append(judgment)
    finally:
        if engine is not None:
            engine_name = stockfish_engine_name(engine)
        if opened_engine and engine is not None:
            engine.quit()

    if skipped_out_of_split:
        logger.warning(
            "Skipped %d prediction row(s) whose example_id is outside the "
            "loaded splits %s",
            skipped_out_of_split,
            split_list,
        )

    prompts_path = output_root / "prompts.jsonl"
    rollouts_path = output_root / "rollouts.jsonl"
    judgments_path = output_root / "judgments.jsonl"
    manifest_path = output_root / "manifest.json"

    write_jsonl(prompts_path, prompts)
    write_jsonl(rollouts_path, rollouts)
    write_jsonl(judgments_path, judgments)

    resolved_splits = _resolved_splits(prompts, split_list)
    manifest = {
        "schema_version": "artifact.v1",
        "artifact_type": "batch_judge_manifest",
        "model_id": model_id,
        "benchmark_dir": str(Path(benchmark_dir)),
        "predictions_path": str(prediction_path),
        "output_dir": str(output_root),
        "splits": resolved_splits,
        "prompt_count": len(prompts),
        "rollout_count": len(rollouts),
        "judgment_count": len(judgments),
        "judge": {
            "mode": "stockfish" if engine is not None else "legality",
            "stockfish_path": str(stockfish_path) if stockfish_path is not None else None,
            "stockfish_depth": stockfish_depth,
            "stockfish_threads": stockfish_threads,
            "stockfish_hash_mb": stockfish_hash_mb,
            "stockfish_engine_name": engine_name,
            "syzygy_path": str(syzygy_path) if syzygy_path is not None else None,
            "chess960": chess960,
        },
        "outputs": {
            "prompts": str(prompts_path),
            "rollouts": str(rollouts_path),
            "judgments": str(judgments_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return BatchJudgeResult(
        prompts_path=prompts_path,
        rollouts_path=rollouts_path,
        judgments_path=judgments_path,
        manifest_path=manifest_path,
        prompt_count=len(prompts),
        rollout_count=len(rollouts),
        judgment_count=len(judgments),
        splits=resolved_splits,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``python -m chess_llm.evals.batch_judge``."""
    parser = argparse.ArgumentParser(
        description="Export benchmark predictions as rollout and judgment artifacts."
    )
    parser.add_argument(
        "--benchmark-dir",
        required=True,
        help="Directory containing frozen benchmark JSONL files.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where artifact JSONL files will be written.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model identifier to store on rollout artifacts.",
    )
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        help="Benchmark split to include. May be repeated.",
    )
    parser.add_argument(
        "--stockfish-path",
        default=None,
        help="Optional Stockfish binary path for engine-backed regret scoring.",
    )
    parser.add_argument(
        "--stockfish-depth",
        type=int,
        default=20,
        help="Stockfish search depth for regret scoring.",
    )
    parser.add_argument(
        "--stockfish-threads",
        type=int,
        default=1,
        help="Stockfish Threads UCI option.",
    )
    parser.add_argument(
        "--stockfish-hash-mb",
        type=int,
        default=256,
        help="Stockfish Hash UCI option in MB.",
    )
    parser.add_argument(
        "--syzygy-path",
        default=None,
        help="Optional Syzygy tablebase directory passed to Stockfish.",
    )
    parser.add_argument(
        "--chess960",
        action="store_true",
        help="Judge FENs and moves as Chess960 positions.",
    )
    args = parser.parse_args(argv)

    judge_prediction_file(
        args.benchmark_dir,
        args.predictions,
        args.output_dir,
        args.model_id,
        splits=args.splits,
        stockfish_path=args.stockfish_path,
        stockfish_depth=args.stockfish_depth,
        stockfish_threads=args.stockfish_threads,
        stockfish_hash_mb=args.stockfish_hash_mb,
        syzygy_path=args.syzygy_path,
        chess960=args.chess960,
    )
    return 0


def _resolved_splits(
    prompts: list[PromptArtifact],
    requested_splits: list[str] | None,
) -> list[str]:
    if requested_splits is not None:
        return sorted(dict.fromkeys(requested_splits))

    found = [
        str(prompt.metadata["split"])
        for prompt in prompts
        if prompt.metadata.get("split") is not None
    ]
    return sorted(dict.fromkeys(found))


def _prompt_chess960(prompt: PromptArtifact, *, default: bool = False) -> bool:
    """Return whether this prompt should be judged with Chess960 rules."""
    benchmark_metadata = prompt.metadata.get("benchmark_metadata")
    if isinstance(benchmark_metadata, dict) and benchmark_metadata.get("is_chess960") is not None:
        return bool(benchmark_metadata["is_chess960"])
    if isinstance(benchmark_metadata, dict) and benchmark_metadata.get("chess960_id") is not None:
        return True
    if prompt.metadata.get("is_chess960") is not None:
        return bool(prompt.metadata["is_chess960"])
    if prompt.metadata.get("chess960_id") is not None:
        return True

    task_type = str(prompt.task_type or prompt.metadata.get("task_type") or "")
    return default or task_type.endswith("_960") or task_type == "chess960"


if __name__ == "__main__":
    raise SystemExit(main())
