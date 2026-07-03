from __future__ import annotations

from chess_llm.evals.trace_metrics import analyze_trace_metrics


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_trace_metrics_measure_referenced_moves_lines_and_conclusion_match():
    prediction = (
        "<think>"
        "Candidates: e2e4 d2d4. "
        "Line: e2e4 e7e5 g1f3. "
        "Backtrack: d2d4 is slower. "
        "Best: e2e4."
        "</think>\n"
        "<move>e2e4</move>"
    )

    metrics = analyze_trace_metrics(STARTING_FEN, prediction)

    assert metrics["trace_referenced_move_accuracy"] == 1.0
    assert metrics["trace_referenced_move_count"] == 7
    assert metrics["trace_illegal_referenced_move_count"] == 0
    assert metrics["trace_candidate_count"] == 2
    assert metrics["trace_line_depth"] == 3
    assert metrics["trace_backtrack_count"] == 1
    assert metrics["trace_step_accuracy"] == 1.0
    assert metrics["trace_conclusion_move_match"] == 1.0


def test_trace_metrics_bucket_illegal_reference_and_move_mismatch():
    prediction = (
        "<think>"
        "Candidates: e2e5 d2d4. "
        "Line: e2e5. "
        "Best: d2d4."
        "</think>\n"
        "<move>e2e4</move>"
    )

    metrics = analyze_trace_metrics(STARTING_FEN, prediction)

    assert metrics["trace_referenced_move_count"] == 4
    assert metrics["trace_illegal_referenced_move_count"] == 2
    assert metrics["trace_referenced_move_accuracy"] == 0.5
    assert metrics["trace_step_accuracy"] == 0.0
    assert metrics["trace_conclusion_move_match"] == 0.0
    assert "illegal_referenced_move" in metrics["trace_failure_buckets"]
    assert "conclusion_move_mismatch" in metrics["trace_failure_buckets"]
