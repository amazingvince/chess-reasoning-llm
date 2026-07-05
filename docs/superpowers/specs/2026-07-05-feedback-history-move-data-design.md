# History-conditioned move-data design

## Feedback item

Add a small amount of history-conditioned move data so the model sees
`PGN prefix + current FEN -> next move`, instead of only bare FEN-to-move rows.

## Scope

- Preserve the UCI move prefix before each extracted Lichess game position.
- Add a Tier 7 generator for next-move imitation from `move_history`, `fen`, and
  `move_played_uci`.
- Keep the assistant target mechanically checkable as exactly
  `<move><uci></move>`.
- Register the generator in default Tier 7 volumes, templates, task metadata,
  identity fields, upload descriptions, and source readiness.

## Data contract

Each eligible `game_positions` row must provide:

- `fen`: current board before the target move.
- `move_history`: space-separated lowercase UCI moves before the target move.
- `move_played_uci`: lowercase UCI target move legal in `fen`.

Rows without history, without a target move, with invalid FEN, or with illegal
target moves are skipped.

## Non-goals

- Do not infer SAN or natural-language PGN history.
- Do not replace Stockfish best-move training data.
- Do not add a new benchmark split in this slice.
