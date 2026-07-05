# Puzzle strata feedback plan

## Task 1: Metadata preservation

Files:

- `src/chess_llm/evals/benchmark.py`
- `tests/test_evals_benchmark.py`

Steps:

- [x] Add a failing test that freezes a puzzle row with top-level rating/themes.
- [x] Preserve `rating` in frozen benchmark metadata.
- [x] Verify the freeze test passes.

## Task 2: Prediction-analysis strata

Files:

- `src/chess_llm/evals/prediction_analysis.py`
- `tests/test_evals_cli.py`

Steps:

- [x] Add a failing CLI test for `puzzle_strata`.
- [x] Accumulate `puzzle_solve` accuracy by rating bucket.
- [x] Accumulate `puzzle_solve` accuracy by theme tag.
- [x] Include missing-rating and missing-theme counts.
- [x] Verify focused CLI test passes.

## Task 3: Verification and commit

Files:

- All files above plus docs.

Steps:

- [x] Run focused puzzle strata tests.
- [x] Run benchmark and prediction-analysis tests.
- [x] Run the full test suite.
- [x] Inspect staged diff.
- [x] Commit the puzzle strata slice.
