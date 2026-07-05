# Feedback Metrics Repair Design

## Goal

Make Phase C move-quality measurements reliable enough to adjudicate checkpoints before launching larger data or SDPO experiments.

## Scope

This first slice covers ACPL and WPD reporting only:

- Clamp every per-example centipawn loss to a fixed upper bound.
- Score malformed, missing, and illegal moves at the same fixed upper bound.
- Report ACPL distribution statistics, not only means.
- Keep WPD visible as the bounded move-quality signal in benchmark reports.

This slice does not change training data generation, curriculum mixing, SDPO rollout construction, artifact mirroring, or run-ledger behavior. Those remain follow-up slices.

## Current Behavior

There are two ACPL implementations:

- `src/chess_llm/evals/run_benchmark.py` scores illegal/missing moves with `max(150, min(abs(gold_cp), 500))`, while legal bad moves can incur mate-scale losses through `mate_score=10000`.
- `src/chess_llm/training/evaluate.py` recomputes a Stockfish baseline and assigns illegal/missing moves a flat `150.0`.

The shared aggregator in `src/chess_llm/evals/benchmark.py` reports only mean ACPL by task and split. This makes a small mate-magnitude tail look like broad move-quality regression, and it can make improved legality look worse because legal moves are exposed to uncapped engine scores.

## Design

Introduce one ACPL policy in `src/chess_llm/evals/benchmark.py`:

- `ACPL_CP_LOSS_CLAMP = 1000.0`
- `ACPL_INVALID_MOVE_PENALTY = ACPL_CP_LOSS_CLAMP`
- `centipawn_loss(gold_cp, predicted_cp, clamp=ACPL_CP_LOSS_CLAMP)` returns `min(max(0, gold_cp - predicted_cp), clamp)`.

Both benchmark entrypoints will import and use `ACPL_INVALID_MOVE_PENALTY`. Illegal, missing, malformed, and unparseable moves receive the clamp value. Legal bad moves also cannot exceed the same clamp after `centipawn_loss`.

Extend `score_split` to report these ACPL fields for every task with ACPL values and for the combined split:

- `<task>_acpl`: arithmetic mean, kept for backward compatibility.
- `<task>_acpl_median`
- `<task>_acpl_p90`
- `<task>_acpl_p95`
- `acpl`
- `acpl_median`
- `acpl_p90`
- `acpl_p95`

Percentiles use nearest-rank interpolation over sorted observed values. For one value, all distribution fields equal that value.

Reports already print keys containing `acpl` in centipawns. No new output section is required for this slice.

## Tests

Add tests before implementation:

- `tests/test_evals_benchmark.py` verifies `centipawn_loss` clamps mate-scale losses and `score_split` emits mean, median, p90, and p95.
- `tests/test_evals_cli.py` verifies standalone `run_benchmark.compute_acpl` assigns the clamp to missing/illegal predictions.
- `tests/test_training_eval_reliability.py` verifies `training.evaluate.compute_acpl` assigns the same clamp to missing/illegal predictions and clamps legal mate-scale losses.

Focused test commands:

```powershell
python -m pytest tests/test_evals_benchmark.py::test_package_centipawn_loss_clamps_tail_values tests/test_evals_benchmark.py::test_package_score_split_reports_acpl_distribution_stats -q
python -m pytest tests/test_evals_cli.py::test_run_benchmark_acpl_invalid_move_uses_shared_clamp -q
python -m pytest tests/test_training_eval_reliability.py::test_compute_acpl_uses_shared_clamp_for_invalid_and_tail_losses -q
```

Full verification:

```powershell
python -m pytest -q
```

## Follow-Up Slices

After this lands:

1. Add a paired comparison CLI for two prediction JSONLs scored on the same frozen examples.
2. Add eval preflight checks for decode budget, benchmark manifest identity, and Stockfish availability.
3. Add loud training harness failures for unsafe resume and multi-GPU non-DDP flash-attention runs.
4. Add run ledger and artifact mirroring.
