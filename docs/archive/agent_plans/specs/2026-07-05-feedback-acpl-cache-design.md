# Feedback ACPL Cache Design

## Goal

Reduce repeated Stockfish work during benchmark scoring by caching scalar ACPL evaluations in the existing SQLite cache sidecar.

## Scope

This slice adds persistent caching for scalar Stockfish evaluations used by ACPL:

- best-position evaluations in `chess-llm-evaluate`,
- predicted post-move evaluations in both `chess-llm-evaluate` and `chess-llm-run-benchmark`.

It reuses the existing `*.multipv.sqlite` file that already stores WPD MultiPV rows. This keeps evaluation artifacts together and lets the run-ledger mirror copy one cache file.

This slice does not add a worker pool or multi-engine parallelization. The cache is the prerequisite throughput improvement with low behavioral risk.

## Cache Key

Scalar entries are keyed by:

- cache schema version,
- evaluation kind (`position` or `post_move`),
- FEN,
- Chess960 flag,
- depth,
- predicted move UCI when applicable,
- POV convention,
- engine configuration payload.

Scores are stored as centipawns using the same mate-score conversion already used by ACPL.

## Integration

`chess_llm.external.multipv.SqliteMultipvCache` owns the new `scalar_evaluations` table beside the existing `multipv_analyses` table.

`chess-llm-evaluate` passes `output.with_suffix(".multipv.sqlite")` into both ACPL and WPD scoring. `chess-llm-run-benchmark` does the same for existing prediction JSONLs.

## Tests

Add focused tests that run ACPL twice against the same cache path:

- the first run uses a fake engine and records expected ACPL,
- the second run uses an engine that raises on analysis,
- the second run must return the cached score without touching the engine.
