# BC Move-Choice Replay Probes

Use this runbook for the feedback-driven 300k-row behavior-cloning probe that
tests how much Phase A/B replay is needed while training move-choice rows.

The trainer now has three named schedules:

| schedule | Tier 7 move-choice rows | Phase A/B replay rows | replay split |
|---|---:|---:|---|
| `bc-probe-r0` | 300,000 | 0 | 0% |
| `bc-probe-r10` | 270,000 | 30,000 | 10% spread evenly over tiers 1-6 |
| `bc-probe-r25` | 225,000 | 75,000 | 25% spread evenly over tiers 1-6 |

## Data Preconditions

- Use a data root whose Tier 7 move-choice rows are benchmark-blocklisted.
- Prefer unique FENs for Tier 7 rows. The intended full run is 1M-5M unique
  move-choice examples; this probe intentionally caps training at 300k rows.
- For canonical generated data, rewrite noisy `7.1`/`7.2` targets with
  `--move-only-task` instead of filtering the whole data root down to those
  tasks. Exclude trace/verifier-only Tier 7 tasks for a pure move-choice
  probe.
- Do not use global `--task-include 7.1...` filters with replay schedules
  unless tiers 1-6 also contain included task names. Global includes can empty
  the replay pools and make the schedule invalid.

## Dry Run

Run each schedule as a dry run first and check the printed segment rows.

```bash
DATA_ROOT=/workspace/chess_sft_data/bc_move_choice/output
OUT_ROOT=/workspace/chess_sft_checkpoints/bc-move-choice-probes
BENCHMARK_DIR=/workspace/chess_sft_data/bc_move_choice/benchmark

chess-llm-train --phase bc-probe-r0 \
  --data-root "$DATA_ROOT" \
  --output-root "$OUT_ROOT" \
  --benchmark-dir "$BENCHMARK_DIR" \
  --schedule-total-examples 300000 \
  --packing off \
  --move-only-task 7.1_best_move_selection \
  --move-only-task 7.2_puzzle_solving \
  --task-exclude 7.9_step_verification \
  --task-exclude 7.10_best_line_trace \
  --dry-run
```

Repeat with `--phase bc-probe-r10` and `--phase bc-probe-r25`.

## Probe Runs

Use one output root per replay ratio or distinct run names under a shared root.
Schedule mode automatically evaluates the full benchmark with soft gates.

```bash
COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --output-root "$OUT_ROOT"
  --benchmark-dir "$BENCHMARK_DIR"
  --schedule-total-examples 300000
  --num-train-epochs 1
  --packing off
  --learning-rate 2e-6
  --trainer-eval-steps 2000
  --trainer-save-steps 2000
  --move-only-task 7.1_best_move_selection
  --move-only-task 7.2_puzzle_solving
  --task-exclude 7.9_step_verification
  --task-exclude 7.10_best_line_trace
  --max-benchmark-examples-per-split 500
  --full-acpl-report
  --run-ledger "$OUT_ROOT/eval_ledger.jsonl"
  --artifact-mirror-dir "$OUT_ROOT/eval_artifacts"
)

chess-llm-train --phase bc-probe-r0 "${COMMON_ARGS[@]}" \
  --run-name bc-probe-r0-300k \
  --wandb-group bc-move-choice-replay \
  --decision-rule "Compare planning WPD/legal/exact move and Phase A/B retention against r10/r25; reject if retention drops outside noise band."

chess-llm-train --phase bc-probe-r10 "${COMMON_ARGS[@]}" \
  --run-name bc-probe-r10-300k \
  --wandb-group bc-move-choice-replay \
  --decision-rule "Prefer if planning improves over r0 without material Phase A/B retention loss; escalate to r25 only if retention regresses."

chess-llm-train --phase bc-probe-r25 "${COMMON_ARGS[@]}" \
  --run-name bc-probe-r25-300k \
  --wandb-group bc-move-choice-replay \
  --decision-rule "Keep only if retention gains justify any planning loss versus r10."
```

## Decision Readout

Compare the three runs with paired prediction comparison where possible. The
primary planning readout is WPD plus legal move rate and exact-move accuracy;
ACPL is secondary and should use median or bounded reporting. Retention readout
comes from the non-planning benchmark splits, especially perception, rules,
tactics, evaluation, openings, and endgames.

Choose the smallest replay ratio that preserves Phase A/B retention while
matching or improving planning WPD/legal move rate. Use that ratio for the
larger 1M-5M unique-FEN BC move-choice run.
