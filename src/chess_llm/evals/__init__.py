"""Evaluation interfaces for static benchmarks and arenas."""

from chess_llm.evals.batch_judge import BatchJudgeResult, judge_prediction_file
from chess_llm.evals.benchmark import (
    BenchmarkExample,
    derive_gold_answer,
    freeze_and_save,
    freeze_split,
    load_benchmark,
    save_benchmark,
    score_prediction,
    score_split,
)
from chess_llm.evals.benchmark_artifacts import (
    benchmark_row_to_prompt,
    load_benchmark_prompts,
)
from chess_llm.evals.eval_harness import EvalResult, evaluate_split
from chess_llm.evals.prediction_analysis import (
    analyze_prediction_file,
    write_prediction_analysis_report,
)

__all__ = [
    "BatchJudgeResult",
    "BenchmarkExample",
    "EvalResult",
    "benchmark_row_to_prompt",
    "derive_gold_answer",
    "evaluate_split",
    "freeze_and_save",
    "freeze_split",
    "judge_prediction_file",
    "load_benchmark",
    "load_benchmark_prompts",
    "save_benchmark",
    "score_prediction",
    "score_split",
    "analyze_prediction_file",
    "write_prediction_analysis_report",
]
