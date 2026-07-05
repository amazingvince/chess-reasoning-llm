# Autodata SFT Refresh

## Purpose

`chess_llm.autodata.sft_refresh` is the first targeted-data builder in the
self-guided loop. It takes artifacts from batch judging and emits ordinary
Tier 7 SFT JSONL rows that the existing training loader can read later.

## Inputs and Outputs

Inputs are the three JSONL files from `chess_llm.evals.batch_judge`:

- `prompts.jsonl`
- `rollouts.jsonl`
- `judgments.jsonl`

Run:

```bash
python -m chess_llm.autodata.sft_refresh \
  --prompts path/to/artifacts/prompts.jsonl \
  --rollouts path/to/artifacts/rollouts.jsonl \
  --judgments path/to/artifacts/judgments.jsonl \
  --output-dir path/to/autodata_refresh
```

Outputs:

- `tier7/7.4_autodata_format_repair.jsonl`
- `tier7/7.5_autodata_move_correction.jsonl`
- `manifest.json`

Each training row uses the package SFT chat schema: `task`, `tier`, `fen`,
`is_chess960`, `messages`, and `metadata`.

## Filters

The builder is deterministic and offline. It only emits examples for
single-move task types: `best_move`, `puzzle_solve`, `endgame_best_move`, and
`tactical_patterns`.

Rows are generated when:

- `parse_failure` has a legal target move.
- `illegal_move` has a legal target move.
- a legal move has `regret_cp >= 100` and a legal teacher move.

Target move precedence is `teacher_move_uci`, then a single legal UCI
`gold_answer`. Rows without valid FENs, valid prompt text, legal target moves,
or supported task types are skipped and counted in the manifest.

## Training Role

This slice does not change the trainer. The refresh files are shaped like
normal Tier 7 data so a later training slice can mix them into a short refresh
run or copy them into a data root.
