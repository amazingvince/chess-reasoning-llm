# Feedback Paired Comparison Design

## Goal

Make checkpoint keep/kill decisions less sensitive to independent-sample noise by comparing two prediction files on the same frozen benchmark examples.

## Scope

This slice adds a lightweight offline comparison command:

- Load one frozen benchmark directory.
- Load two existing prediction JSONLs.
- Score both prediction files on the same examples with the existing benchmark scorers.
- Report paired correct/incorrect counts and exact McNemar p-values.
- Write the same report to JSON when requested.

This slice does not run model inference, compute Stockfish ACPL/WPD, change training, or decide pass/fail thresholds. It provides the paired statistics needed before those decisions are made.

## CLI

```powershell
chess-llm-compare-predictions `
  --benchmark-dir benchmark/frozen `
  --baseline artifacts/evals/repair2/predictions.jsonl `
  --candidate artifacts/evals/moveonly3/predictions.jsonl `
  --output-json artifacts/evals/repair2-vs-moveonly3.comparison.json
```

The command reports:

- `baseline_accuracy`
- `candidate_accuracy`
- `delta`
- `baseline_mean_score`
- `candidate_mean_score`
- `mean_score_delta`
- `baseline_only`
- `candidate_only`
- `both_correct`
- `both_wrong`
- `mcnemar_p`

Results are emitted for the overall benchmark, each split, and each `split/task_type` pair.

## Statistical Policy

McNemar's exact test uses only discordant binary outcomes:

- `baseline_only`: baseline scored exactly `1.0`, candidate did not.
- `candidate_only`: candidate scored exactly `1.0`, baseline did not.

The p-value is the exact two-sided binomial tail under p=0.5. Continuous primary scores are still reported as mean-score deltas, but the McNemar p-value is intentionally attached to the perfect-hit binary view.

## Tests

Add focused tests in `tests/test_evals_compare_predictions.py`:

- exact McNemar p-value math,
- paired counts over a synthetic frozen planning benchmark,
- CLI print path and JSON output path.
