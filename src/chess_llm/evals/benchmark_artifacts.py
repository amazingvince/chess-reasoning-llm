"""Adapters from frozen benchmark JSONL rows to prompt artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any

from chess_llm.artifacts.schemas import ChatMessage, PromptArtifact


def benchmark_row_to_prompt(
    row: dict[str, Any],
    source_path: str | Path | None = None,
) -> PromptArtifact:
    """Convert one frozen benchmark row into a prompt artifact."""
    missing = [key for key in ("example_id", "prompt") if key not in row]
    if missing:
        raise ValueError(f"benchmark row missing required field(s): {missing}")

    metadata = {
        "split": row.get("split"),
        "task_type": row.get("task_type"),
        "gold_answer": row.get("gold_answer"),
        "metric_type": row.get("metric_type"),
        "source_benchmark_path": str(source_path) if source_path is not None else None,
        "benchmark_metadata": dict(row.get("metadata") or {}),
    }

    return PromptArtifact(
        prompt_id=str(row["example_id"]),
        messages=[ChatMessage(role="user", content=str(row["prompt"]))],
        fen=row.get("fen"),
        task_type=row.get("task_type"),
        metadata=metadata,
    )


def load_benchmark_prompts(
    benchmark_dir: str | Path,
    splits: Iterable[str] | None = None,
) -> dict[str, PromptArtifact]:
    """Load frozen benchmark JSONL rows as prompt artifacts keyed by ID."""
    root = Path(benchmark_dir)
    split_filter = set(splits) if splits is not None else None
    prompts: dict[str, PromptArtifact] = {}

    for path in sorted(root.glob("*.jsonl")):
        if path.name == "manifest.json":
            continue
        with path.open(encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if split_filter is not None and row.get("split") not in split_filter:
                    continue
                prompt = benchmark_row_to_prompt(row, source_path=path)
                if prompt.prompt_id in prompts:
                    raise ValueError(
                        f"duplicate benchmark example_id {prompt.prompt_id!r} "
                        f"in {path} line {line_number}"
                    )
                prompts[prompt.prompt_id] = prompt

    return prompts
