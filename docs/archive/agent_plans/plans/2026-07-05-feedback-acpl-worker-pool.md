# ACPL worker-pool feedback plan

## Task 1: Red tests

Files:

- `tests/test_training_eval_reliability.py`
- `tests/test_evals_cli.py`

Steps:

- [x] Add failing tests for `compute_acpl(..., workers=N, engine_factory=...)`
  in training eval and standalone benchmark scoring.
- [x] Add failing tests for `--stockfish-workers` parser/config plumbing.

## Task 2: Worker implementation

Files:

- `src/chess_llm/training/evaluate.py`
- `src/chess_llm/evals/run_benchmark.py`

Steps:

- [x] Represent scalar ACPL cache misses as evaluation jobs.
- [x] Resolve scalar cache hits before worker execution.
- [x] Split cache misses across per-worker Stockfish engines.
- [x] Write scalar cache results back on the main thread.
- [x] Preserve default single-worker behavior.

## Task 3: Verification and commit

Files:

- All files above plus docs.

Steps:

- [x] Run focused worker-pool tests.
- [x] Run broader eval/cache/artifact tests.
- [x] Run the full test suite.
- [x] Inspect staged diff.
- [x] Commit the ACPL worker-pool slice.
