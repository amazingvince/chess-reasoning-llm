"""Build legacy SFT refresh rows from judged Autodata artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from chess_llm.artifacts.jsonl import read_jsonl
from chess_llm.artifacts.schemas import JudgmentArtifact, PromptArtifact, RolloutArtifact
from chess_llm.autodata.failure_buckets import ILLEGAL_MOVE, LEGAL_UNSCORED, MISSING_FEN, PARSE_FAILURE
from chess_llm.core.board import is_legal_move, validate_fen, variant_fen_key
from chess_llm.sft import build_sft_row


FORMAT_REPAIR_TASK = "7.4_autodata_format_repair"
MOVE_CORRECTION_TASK = "7.5_autodata_move_correction"
DEFAULT_MOVE_TASK_TYPES = frozenset(
    {"best_move", "puzzle_solve", "endgame_best_move", "tactical_patterns"}
)
_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


@dataclass(frozen=True)
class SftRefreshResult:
    """Paths and counts produced by one SFT refresh build."""

    format_repair_path: Path
    move_correction_path: Path
    manifest_path: Path
    format_repair_count: int
    move_correction_count: int
    skipped_count: int
    skip_reasons: dict[str, int]


def build_sft_refresh(
    prompts_path: str | Path,
    rollouts_path: str | Path,
    judgments_path: str | Path,
    output_dir: str | Path,
    *,
    min_regret_cp: float = 100.0,
    task_types: set[str] | list[str] | tuple[str, ...] | None = None,
    max_examples: int | None = None,
) -> SftRefreshResult:
    """Convert judged rollout artifacts into legacy-compatible Tier 7 rows."""
    prompts = _index_prompts(prompts_path)
    rollouts = _index_rollouts(rollouts_path)
    allowed_task_types = set(task_types) if task_types is not None else set(DEFAULT_MOVE_TASK_TYPES)

    format_rows: list[dict] = []
    correction_rows: list[dict] = []
    skip_reasons: Counter[str] = Counter()
    seen_keys: set[tuple[str, str, str]] = set()

    for judgment in read_jsonl(judgments_path, JudgmentArtifact):
        if max_examples is not None and len(format_rows) + len(correction_rows) >= max_examples:
            skip_reasons["max_examples"] += 1
            continue

        rollout = rollouts.get(judgment.rollout_id)
        if rollout is None:
            skip_reasons["missing_rollout"] += 1
            continue
        prompt = prompts.get(rollout.prompt_id)
        if prompt is None:
            skip_reasons["missing_prompt"] += 1
            continue

        row_task = _refresh_task_for_judgment(judgment, min_regret_cp=min_regret_cp)
        if row_task is None:
            skip_reasons[_skip_reason_for_judgment(judgment, min_regret_cp=min_regret_cp)] += 1
            continue

        task_type = str(prompt.task_type or prompt.metadata.get("task_type") or "")
        if task_type not in allowed_task_types:
            skip_reasons["non_move_task"] += 1
            continue

        fen = prompt.fen
        if not fen:
            skip_reasons["missing_fen"] += 1
            continue
        chess960 = _is_chess960(prompt)
        if not validate_fen(fen, chess960=chess960):
            skip_reasons["invalid_fen"] += 1
            continue

        user_prompt = _user_prompt(prompt)
        if user_prompt is None:
            skip_reasons["missing_prompt_text"] += 1
            continue

        target_move, target_source = _target_move(prompt, judgment, fen=fen, chess960=chess960)
        if target_move is None or _requires_teacher_target(judgment, row_task, target_source):
            skip_reasons["missing_target_move"] += 1
            continue

        dedupe_key = (row_task, variant_fen_key(fen, chess960=chess960), target_move)
        if dedupe_key in seen_keys:
            skip_reasons["duplicate"] += 1
            continue
        seen_keys.add(dedupe_key)

        row = _build_training_row(
            row_task,
            prompt,
            rollout,
            judgment,
            fen=fen,
            user_prompt=user_prompt,
            target_move=target_move,
            target_source=target_source,
            chess960=chess960,
            prompts_path=Path(prompts_path),
            rollouts_path=Path(rollouts_path),
            judgments_path=Path(judgments_path),
        )
        if row_task == FORMAT_REPAIR_TASK:
            format_rows.append(row)
        else:
            correction_rows.append(row)

    output_root = Path(output_dir)
    tier7_dir = output_root / "tier7"
    format_path = tier7_dir / f"{FORMAT_REPAIR_TASK}.jsonl"
    correction_path = tier7_dir / f"{MOVE_CORRECTION_TASK}.jsonl"
    manifest_path = output_root / "manifest.json"

    _write_legacy_jsonl(format_path, format_rows)
    _write_legacy_jsonl(correction_path, correction_rows)

    manifest = {
        "schema_version": "artifact.v1",
        "artifact_type": "sft_refresh_manifest",
        "prompts_path": str(Path(prompts_path)),
        "rollouts_path": str(Path(rollouts_path)),
        "judgments_path": str(Path(judgments_path)),
        "output_dir": str(output_root),
        "min_regret_cp": float(min_regret_cp),
        "task_types": sorted(allowed_task_types),
        "max_examples": max_examples,
        "format_repair_count": len(format_rows),
        "move_correction_count": len(correction_rows),
        "skipped_count": sum(skip_reasons.values()),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "outputs": {
            "format_repair": str(format_path),
            "move_correction": str(correction_path),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return SftRefreshResult(
        format_repair_path=format_path,
        move_correction_path=correction_path,
        manifest_path=manifest_path,
        format_repair_count=len(format_rows),
        move_correction_count=len(correction_rows),
        skipped_count=sum(skip_reasons.values()),
        skip_reasons=dict(skip_reasons),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``python -m chess_llm.autodata.sft_refresh``."""
    parser = argparse.ArgumentParser(
        description="Build Tier 7 SFT refresh rows from judged Autodata artifacts."
    )
    parser.add_argument("--prompts", required=True, help="Path to prompts.jsonl.")
    parser.add_argument("--rollouts", required=True, help="Path to rollouts.jsonl.")
    parser.add_argument("--judgments", required=True, help="Path to judgments.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory for refresh JSONL output.")
    parser.add_argument(
        "--min-regret-cp",
        type=float,
        default=100.0,
        help="Minimum Stockfish regret in centipawns for legal move correction rows.",
    )
    parser.add_argument(
        "--task-type",
        action="append",
        dest="task_types",
        help="Single-move benchmark task type to include. May be repeated.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap across all emitted refresh rows.",
    )
    args = parser.parse_args(argv)

    build_sft_refresh(
        args.prompts,
        args.rollouts,
        args.judgments,
        args.output_dir,
        min_regret_cp=args.min_regret_cp,
        task_types=args.task_types,
        max_examples=args.max_examples,
    )
    return 0


def _index_prompts(path: str | Path) -> dict[str, PromptArtifact]:
    prompts: dict[str, PromptArtifact] = {}
    for prompt in read_jsonl(path, PromptArtifact):
        if prompt.prompt_id in prompts:
            raise ValueError(f"duplicate prompt_id {prompt.prompt_id!r}")
        prompts[prompt.prompt_id] = prompt
    return prompts


def _index_rollouts(path: str | Path) -> dict[str, RolloutArtifact]:
    rollouts: dict[str, RolloutArtifact] = {}
    for rollout in read_jsonl(path, RolloutArtifact):
        if rollout.rollout_id in rollouts:
            raise ValueError(f"duplicate rollout_id {rollout.rollout_id!r}")
        rollouts[rollout.rollout_id] = rollout
    return rollouts


def _refresh_task_for_judgment(
    judgment: JudgmentArtifact,
    *,
    min_regret_cp: float,
) -> str | None:
    if judgment.failure_bucket == PARSE_FAILURE:
        return FORMAT_REPAIR_TASK
    if judgment.failure_bucket == ILLEGAL_MOVE:
        return MOVE_CORRECTION_TASK
    if judgment.legal is True and judgment.regret_cp is not None and judgment.regret_cp >= min_regret_cp:
        return MOVE_CORRECTION_TASK
    return None


def _skip_reason_for_judgment(
    judgment: JudgmentArtifact,
    *,
    min_regret_cp: float,
) -> str:
    if judgment.failure_bucket == MISSING_FEN:
        return "missing_fen"
    if judgment.failure_bucket == LEGAL_UNSCORED:
        return "legal_unscored"
    if judgment.legal is True:
        if judgment.regret_cp is None:
            return "unscored_legal"
        if judgment.regret_cp < min_regret_cp:
            return "low_regret"
    return "unsupported_judgment"


def _target_move(
    prompt: PromptArtifact,
    judgment: JudgmentArtifact,
    *,
    fen: str,
    chess960: bool,
) -> tuple[str | None, str | None]:
    teacher_move = _valid_target(judgment.teacher_move_uci, fen=fen, chess960=chess960)
    if teacher_move is not None:
        return teacher_move, "teacher_move_uci"

    gold_move = _single_gold_move(prompt.metadata.get("gold_answer"))
    gold_move = _valid_target(gold_move, fen=fen, chess960=chess960)
    if gold_move is not None:
        return gold_move, "gold_answer"

    return None, None


def _requires_teacher_target(
    judgment: JudgmentArtifact,
    row_task: str,
    target_source: str | None,
) -> bool:
    return (
        row_task == MOVE_CORRECTION_TASK
        and judgment.legal is True
        and judgment.failure_bucket is None
        and target_source != "teacher_move_uci"
    )


def _single_gold_move(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip().lower()
    if _UCI_RE.fullmatch(stripped):
        return stripped
    return None


def _valid_target(move_uci: str | None, *, fen: str, chess960: bool) -> str | None:
    if move_uci is None:
        return None
    normalized = move_uci.strip().lower()
    if not _UCI_RE.fullmatch(normalized):
        return None
    if not is_legal_move(fen, normalized, chess960=chess960):
        return None
    return normalized


def _user_prompt(prompt: PromptArtifact) -> str | None:
    for message in prompt.messages:
        if message.role == "user" and message.content.strip():
            return message.content
    return None


def _is_chess960(prompt: PromptArtifact) -> bool:
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
    return task_type.endswith("_960") or task_type == "chess960"


def _build_training_row(
    task: str,
    prompt: PromptArtifact,
    rollout: RolloutArtifact,
    judgment: JudgmentArtifact,
    *,
    fen: str,
    user_prompt: str,
    target_move: str,
    target_source: str,
    chess960: bool,
    prompts_path: Path,
    rollouts_path: Path,
    judgments_path: Path,
) -> dict:
    benchmark_metadata = prompt.metadata.get("benchmark_metadata")
    chess960_id = prompt.metadata.get("chess960_id")
    if chess960_id is None and isinstance(benchmark_metadata, dict):
        chess960_id = benchmark_metadata.get("chess960_id")
    metadata = {
        "source": "autodata_sft_refresh",
        "source_prompt_id": prompt.prompt_id,
        "source_rollout_id": rollout.rollout_id,
        "source_judgment_id": judgment.judgment_id,
        "model_id": rollout.model_id,
        "failure_bucket": judgment.failure_bucket,
        "regret_cp": judgment.regret_cp,
        "parsed_model_move_uci": rollout.parsed_answer.move_uci,
        "target_move_uci": target_move,
        "target_source": target_source,
        "source_prompts_path": str(prompts_path),
        "source_rollouts_path": str(rollouts_path),
        "source_judgments_path": str(judgments_path),
    }
    if chess960_id is not None:
        metadata["chess960_id"] = chess960_id
    return build_sft_row(
        task=task,
        tier=7,
        fen=fen,
        is_chess960=chess960,
        user_prompt=user_prompt,
        assistant_content=_assistant_content(task, target_move),
        metadata=metadata,
    )


def _assistant_content(task: str, target_move: str) -> str:
    if task == FORMAT_REPAIR_TASK:
        thought = (
            "The previous answer did not provide a single parseable UCI move. "
            "Use the verified target move."
        )
    else:
        thought = (
            "The previous move was not the verified target for this position. "
            "Use the corrected move."
        )
    return f"<think>{thought}</think>\n<move>{target_move}</move>"


def _write_legacy_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
