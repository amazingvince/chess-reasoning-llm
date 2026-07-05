# Static exchange evaluation design

## Feedback item

Add the next mechanical primitive after attacker/defender counting: static
exchange evaluation for one capture, including the recapture sequence and net
material result.

## Scope

- Add a Tier 3 task `3.9_static_exchange_evaluation`.
- Use existing `fen_pool` positions and select legal captures.
- Skip en-passant and promotion captures in this slice to keep the grammar and
  material accounting mechanical.
- Follow a deterministic least-valuable-attacker recapture policy.
- Emit a fixed four-line answer with move, target square, capture sequence, and
  net material in pawns for the original side to move.
- Store capture move, target square, sequence, and net material in metadata.

## Answer grammar

```text
Move: <uci>
Target square: <square>
Capture sequence: <uci>[, <uci>...]
Net material for <white|black>: <signed-int> pawns
```

## Non-goals

- Do not add a benchmark split in this slice.
- Do not use Stockfish or engine eval.
- Do not implement full minimax SEE with promotion/en-passant edge cases yet.
