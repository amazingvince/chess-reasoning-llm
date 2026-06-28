import json
from pathlib import Path

from chess_llm.artifacts.schemas import PromptArtifact
from chess_llm.evals.benchmark_artifacts import (
    benchmark_row_to_prompt,
    load_benchmark_prompts,
)


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _benchmark_row(example_id: str, split: str = "planning") -> dict:
    return {
        "example_id": example_id,
        "split": split,
        "task_type": "best_move",
        "fen": STARTING_FEN,
        "prompt": f"FEN: {STARTING_FEN}\nWhat is the best move?",
        "gold_answer": "e2e4",
        "metric_type": "move_extraction",
        "metadata": {"source": "unit"},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_benchmark_row_to_prompt_preserves_benchmark_fields():
    source_path = Path("benchmark") / "planning.jsonl"
    row = _benchmark_row("planning_00000")

    prompt = benchmark_row_to_prompt(row, source_path=source_path)

    assert isinstance(prompt, PromptArtifact)
    assert prompt.prompt_id == "planning_00000"
    assert prompt.messages[0].role == "user"
    assert prompt.messages[0].content == row["prompt"]
    assert prompt.fen == STARTING_FEN
    assert prompt.task_type == "best_move"
    assert prompt.metadata["split"] == "planning"
    assert prompt.metadata["task_type"] == "best_move"
    assert prompt.metadata["gold_answer"] == "e2e4"
    assert prompt.metadata["metric_type"] == "move_extraction"
    assert prompt.metadata["source_benchmark_path"] == str(source_path)
    assert prompt.metadata["benchmark_metadata"] == {"source": "unit"}


def test_load_benchmark_prompts_reads_all_split_jsonl_files(tmp_path):
    _write_jsonl(tmp_path / "planning.jsonl", [_benchmark_row("planning_00000")])
    _write_jsonl(tmp_path / "rules.jsonl", [_benchmark_row("rules_00000", "rules")])
    (tmp_path / "manifest.json").write_text('{"splits": {"planning": 1}}')

    prompts = load_benchmark_prompts(tmp_path)

    assert list(prompts) == ["planning_00000", "rules_00000"]
    assert prompts["rules_00000"].metadata["source_benchmark_path"].endswith(
        "rules.jsonl"
    )


def test_load_benchmark_prompts_filters_splits(tmp_path):
    _write_jsonl(tmp_path / "planning.jsonl", [_benchmark_row("planning_00000")])
    _write_jsonl(tmp_path / "rules.jsonl", [_benchmark_row("rules_00000", "rules")])

    prompts = load_benchmark_prompts(tmp_path, splits=["rules"])

    assert list(prompts) == ["rules_00000"]
