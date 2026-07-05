# Self-play reliability feedback plan

## Task 1: Red tests

Files:

- `tests/test_autodata_selfplay.py`

Steps:

- [x] Assert all-legal self-play manifests report 100% legal and clean games.
- [x] Assert failed self-play games report parse/illegal buckets and fallbacks.

## Task 2: Reliability metrics

Files:

- `src/chess_llm/autodata/selfplay.py`

Steps:

- [x] Add per-game model-turn, legal-turn, failure, and fallback counts.
- [x] Add aggregate manifest reliability metrics.
- [x] Keep the metrics available for legality-only and Stockfish-judged runs.

## Task 3: Verification and commit

Files:

- All files above plus docs.

Steps:

- [x] Run focused self-play reliability tests.
- [x] Run broader autodata tests.
- [x] Run the full test suite.
- [x] Inspect staged diff.
- [x] Commit the self-play reliability slice.
