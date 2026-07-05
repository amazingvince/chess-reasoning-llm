# Feedback Eval Preflight Design

## Goal

Make the evaluation harness fail before model loading for known silent failure modes called out in the feedback.

## Scope

This slice adds preflight checks inside `chess-llm-evaluate` after benchmark split loading and before W&B/model initialization:

- Phase C planning trace evals require a configurable minimum decode budget.
- Required ACPL/WPD scoring requires a resolvable Stockfish binary.
- Benchmark manifest enforcement is available through an explicit strict flag.

This slice does not add disk-headroom checks, GPU-order checks, contamination checks, or artifact mirroring. Those remain follow-up harness/ops work.

## Policy

The default decode budget remains `--max-new-tokens 512`. Phase C planning trace evals fail preflight when `--max-new-tokens` is below `--min-planning-max-new-tokens`, default `256`.

ACPL/WPD preflight is strict only when scoring would require Stockfish:

- Phase C planning evals, unless `--no-acpl` is set.
- Any run using `--full-acpl-report`, unless `--no-acpl` is set.

Older local benchmarks without manifests remain usable by default. H100 or release runs can opt into strict identity checks with `--require-benchmark-manifest`, which fails when `manifest.json` is missing or lacks `version`.

Preflight failures return `EVAL_INFRA_FAILURE_EXIT_CODE` and write the normal `*.eval_run.json` artifact with `metadata.error = "eval_preflight_failed"` and the message list.

## Tests

Add focused tests in `tests/test_training_eval_reliability.py`:

- Phase C planning traces with too-small decode budget fail before model load.
- Required Phase C ACPL with missing Stockfish fails before model load.
- `--require-benchmark-manifest` fails when `manifest.json` is absent.
