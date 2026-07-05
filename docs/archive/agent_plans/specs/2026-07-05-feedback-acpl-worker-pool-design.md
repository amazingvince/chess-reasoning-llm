# ACPL worker-pool design

## Feedback item

After adding the scalar ACPL cache, parallelize Stockfish scoring so larger
planning and BC-scale evaluations do not bottleneck on one `Threads=1` engine.

## Scope

- Keep `workers=1` as the default behavior.
- Add `--stockfish-workers` to `chess-llm-evaluate` and
  `chess-llm-run-benchmark`.
- Use worker engines only for scalar ACPL cache misses.
- Keep SQLite cache reads and writes on the main thread; workers return scalar
  centipawn results only.
- Preserve the existing WPD path in this slice.

## Execution model

The ACPL scorer validates move tags and legality first, then builds scalar
evaluation jobs:

- `position`: side-to-move best-play baseline for training eval ACPL.
- `post_move`: predicted move result from original side-to-move perspective.

Cache hits are resolved before any worker starts. Misses are split across
worker chunks, each chunk opens one Stockfish engine, evaluates its assigned
jobs sequentially, and closes the engine.

## Non-goals

- Do not parallelize WPD/MultiPV in this slice.
- Do not change ACPL scoring semantics.
- Do not change cache keys or invalidate existing cache rows.
