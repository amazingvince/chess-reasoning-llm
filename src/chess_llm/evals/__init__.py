"""Evaluation interfaces for static benchmarks and arenas."""

from chess_llm.evals.batch_judge import BatchJudgeResult, judge_prediction_file
from chess_llm.evals.benchmark_artifacts import (
    benchmark_row_to_prompt,
    load_benchmark_prompts,
)

__all__ = [
    "BatchJudgeResult",
    "benchmark_row_to_prompt",
    "judge_prediction_file",
    "load_benchmark_prompts",
]
