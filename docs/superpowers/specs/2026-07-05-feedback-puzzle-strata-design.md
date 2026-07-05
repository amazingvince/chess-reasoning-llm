# Puzzle strata eval design

## Feedback item

Puzzle pass rate should be broken down by Lichess rating bucket and theme tag so
planning changes show a capability curve instead of a single blended number.

## Scope

- Preserve top-level puzzle `rating` in frozen benchmark metadata.
- Reuse existing puzzle `themes` metadata.
- Add a `puzzle_strata` section to prediction analysis JSON.
- Report row count, example count, scored count, and primary accuracy for each
  rating bucket and theme.

## Rating buckets

Use stable half-open buckets:

- `<800`
- `800-1199`
- `1200-1599`
- `1600-1999`
- `2000-2399`
- `2400-2799`
- `2800+`

## Non-goals

- Do not change benchmark scoring.
- Do not add a new training task.
- Do not require Stockfish for this analysis.
