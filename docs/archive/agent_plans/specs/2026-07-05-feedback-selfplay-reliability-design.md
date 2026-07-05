# Self-play reliability eval design

## Feedback item

Self-play needs a game-level legality/reliability signal, not only per-prompt
benchmark legal move rate. A model can look acceptable per move while still
failing too often across full games.

## Scope

- Reuse the existing self-play driver and artifact outputs.
- Add per-game model-turn and failure counts to `games.jsonl`.
- Add aggregate reliability metrics to the self-play `manifest.json`.
- Keep the metric independent of Stockfish scoring; legality-only runs should
  still report reliability.

## Metrics

- `model_turn_count`: model outputs judged during the run.
- `legal_model_turn_count`: judged model outputs with `legal is True`.
- `model_failure_count`: judged model outputs that were not legal.
- `legal_model_turn_rate`: legal model turns divided by model turns.
- `clean_game_count`: games with zero model failures.
- `clean_game_rate`: clean games divided by total games.
- `random_fallback_count`: fallback moves inserted after invalid model output.
- `failure_buckets`: parse/illegal/etc. counts for failed model turns.

## Non-goals

- Do not change self-play move selection.
- Do not change fallback behavior.
- Do not add Stockfish-dependent pass/fail gates in this slice.
