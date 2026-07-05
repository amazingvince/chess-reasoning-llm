# 2026-07-05 Phase C Contract-Fix Probe

## Objective

Continue Phase C despite the failed Phase B gate, but make the measurement
surface honest first. The previous Phase C probe showed near-zero
`format_compliance`; investigation found this was partly a benchmark/evaluator
problem, not only a model problem.

## Root Cause

Two issues were fixed before rerunning:

- Planning benchmark prompts for `best_move` and `puzzle_solve` did not ask for
  `<think>...</think><move>...</move>`, while the Phase C gate required that
  format.
- The planning eval source appended MultiPV rows after duplicate puzzle/best
  move FENs, so FEN dedup dropped all `candidate_ratings`, `best_line_trace`,
  and `step_verification` benchmark rows.

A third evaluator issue was found after the run:

- Qwen-style chat templates prefill assistant thinking tokens. Evaluation was
  decoding only new tokens, so strict protocol metrics missed the implicit
  `<think>` prefix. Eval now enables thinking only for planning trace-protocol
  tasks and stitches the assistant prefill back into raw predictions before
  scoring.

## Local Code Changes

- Added answer contracts for:
  - `best_move`
  - `puzzle_solve`
  - `7.1_best_move_selection`
  - `7.2_puzzle_solving`
  - `7.3_move_consequence`
- Moved cached MultiPV rows ahead of older planning sources and distributed
  them across `candidate_ratings`, `best_line_trace`, and
  `step_verification`.
- Added prefill-aware eval scoring for Qwen thinking templates.

Verification:

```text
python -m pytest tests/test_training_eval_reliability.py tests/test_evals_benchmark.py tests/test_sft_pipeline.py::test_eval_split_sources_include_multipv_planning_tasks -q
138 passed in 1.59s
```

## Data Regeneration

Remote data root:

```text
/workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703
```

Regenerated Tier 7 and refreshed benchmark splits. Final planning benchmark
coverage:

| task | count |
|---|---:|
| best_move | 767 |
| puzzle_solve | 1089 |
| candidate_ratings | 33 |
| best_line_trace | 69 |
| step_verification | 42 |

All `best_move` and `puzzle_solve` prompts had the `<think>/<move>` contract.
Tier 7 source files were regenerated with the same contracts.

## Run

Run root:

```text
/workspace/chess_sft_checkpoints/phase-c-v2000-contractfix-repair2-h100-fa3-20260705
```

Starting checkpoint:

```text
/workspace/chess_sft_checkpoints/phase-b-repair2-t3t6-v50000-h100-fa3-20260704/phase_b/best
```

Training recipe:

- Phase: `c`
- Attention: `flash_attention_3`
- Packing: `auto`
- Steps: `500`
- LR: `2e-6`
- Max train examples: `220000`
- Phase C mixer used with stronger task upsampling:
  - `7.1_best_move_selection=8`
  - `7.2_puzzle_solving=8`
  - `7.3_move_consequence=4`
  - `7.8_candidate_ratings=6`
  - `7.10_best_line_trace=6`
  - `7.9_step_verification=4`

Training metrics:

```text
train_loss: 0.19159470796585082
num_input_tokens_seen: 31,786,816
train_runtime: 619.2392s
train_tokens_per_second: 51,332
```

## Prefill-Aware Eval Results

500 examples per split, report-only, Stockfish depth 12 for planning ACPL.

| metric | repair2 pre | contractfix post |
|---|---:|---:|
| planning overall | 6.2% | 15.5% |
| format_compliance | 0.0% | 99.6% |
| legal_move_rate | 0.0% | 63.1% |
| missing_move_tag_count | 471 | 0 |
| best_move | 1.7% | 6.9% |
| puzzle_solve | 10.0% | 14.3% |
| candidate_ratings | 0.0% | 72.2% |
| candidate best match | 0.0% | 100.0% |
| candidate bucket accuracy | 0.0% | 16.7% |
| best_line_trace | 0.0% | 0.0% |
| best_line format compliance | 0.0% | 100.0% |
| step_verification | 0.0% | 100.0% |
| WPD | 48.4% | 47.7% |
| ACPL | 725.4 cp | 1539.5 cp |

Non-planning regression:

| split | repair2 pre | contractfix post |
|---|---:|---:|
| perception | 99.2% | 99.2% |
| rules | 86.8% | 85.1% |
| tactics | 46.4% | 46.1% |
| evaluation | 84.8% | 84.2% |
| openings | 54.5% | 54.0% |
| endgames | 70.4% | 71.4% |

## Interpretation

This run succeeded as a format and verifier/candidate-rating repair:

- The model now reliably emits move tags under the corrected protocol.
- `candidate_ratings` learned candidate-set and best-move structure quickly.
- `step_verification` learned the mechanically corrupted candidate-rating judge
  task immediately.

It did not improve chess move quality enough:

- Legal move rate is only `63.1%`.
- ACPL remains far above the Phase C target.
- `best_line_trace` is format-correct but not faithful to the cached engine PV.
- Candidate bucket values are structurally formatted but poorly calibrated.

## Next Iteration

This was tried as `movequality1`.

Run root:

```text
/workspace/chess_sft_checkpoints/phase-c-v2000-movequality1-from-contractfix-h100-fa3-20260705
```

Starting checkpoint:

```text
/workspace/chess_sft_checkpoints/phase-c-v2000-contractfix-repair2-h100-fa3-20260705/phase_c/best
```

Training recipe:

- Phase: `c`
- Attention: `flash_attention_3`
- Packing: `auto`
- Steps: `300`
- LR: `1e-6`
- Max train examples: `220000`
- Up-sampled only:
  - `7.1_best_move_selection=10`
  - `7.2_puzzle_solving=10`
  - `7.3_move_consequence=4`

Training metrics:

```text
train_loss: 0.07572362899780273
num_input_tokens_seen: 19,080,848
train_runtime: 381.5335s
train_tokens_per_second: about 50,010-50,785
```

Prefill-aware eval results, 500 examples per split:

| metric | contractfix | movequality1 |
|---|---:|---:|
| planning overall | 15.5% | 14.9% |
| format_compliance | 99.6% | 100.0% |
| legal_move_rate | 63.1% | 60.1% |
| missing_move_tag_count | 0 | 0 |
| best_move | 6.9% | 6.3% |
| puzzle_solve | 14.3% | 13.2% |
| candidate_ratings | 72.2% | 81.1% |
| candidate best match | 100.0% | 100.0% |
| candidate bucket accuracy | 16.7% | 43.3% |
| best_line_trace | 0.0% | 0.0% |
| best_line format compliance | 100.0% | 100.0% |
| step_verification | 100.0% | 100.0% |
| WPD | 47.7% | 49.3% |
| ACPL | 1539.5 cp | 1535.3 cp |

Non-planning comparison:

| split | contractfix | movequality1 |
|---|---:|---:|
| perception | 99.2% | 99.2% |
| rules | 85.1% | 85.5% |
| tactics | 46.1% | 46.5% |
| evaluation | 84.2% | 84.2% |
| openings | 54.0% | 54.2% |
| endgames | 71.4% | 70.8% |

Decision:

- Do not continue from `movequality1`.
- Keep `contractfix` as the useful protocol/format checkpoint.
- Keep Phase B `repair2` as the stronger raw generator checkpoint.
- R1/R2/R3 remain blocked until trace faithfulness metrics can distinguish
  formatted but unfaithful traces from engine-faithful traces.

Next useful work:

- Run a small supervised overfit/memorization sanity check on `7.1`/`7.2` only
  to see whether this model can learn exact move selection under the current
  template.
- If that passes, build a narrower Phase C mixer with much stronger best-move,
  puzzle, and tactics replay instead of relying on broad Phase C continuation.
- If that fails, fix the task representation before spending more H100 time.

Do not mark Phase C as passed. These runs repaired measurement and output
protocol, but they did not yet produce a chess-quality planning model.

## 2026-07-05 Follow-Up Diagnostics

Two targeted diagnostics were run after `movequality1`.

### Overfit7x2

Purpose: test whether the current `7.1`/`7.2` task representation and
`<think>/<move>` template are learnable at all.

Run root:

```text
/workspace/chess_sft_checkpoints/phase-c-overfit7x2-from-contractfix-h100-fa3-20260705T091237Z
```

Data root:

```text
/workspace/chess_sft_data/phase-c-overfit7x2-20260705T091237Z
```

Setup:

- Start checkpoint: `contractfix` best.
- Mini train/eval set: 128 `7.1_best_move_selection` rows and 128
  `7.2_puzzle_solving` rows, plus tiny filler for hard-wired Phase C tiers.
- Training: 200 steps, LR `1e-5`, FA3, packing auto.
- Eval: same 256 examples converted into a planning mini benchmark.

Results:

| metric | contractfix pre | overfit post |
|---|---:|---:|
| planning overall | 28.5% | 98.8% |
| best_move | 23.4% | 98.4% |
| puzzle_solve | 33.6% | 99.2% |
| format_compliance | 100.0% | 100.0% |
| legal_move_rate | 76.2% | 100.0% |
| missing_move_tag_count | 0 | 0 |

Training metrics:

```text
train_loss: 0.004851598741661292
num_input_tokens_seen: 12,618,400
train_runtime: 256.0503s
train_tokens_per_second: 49,280.9
```

Decision: the task/template can be learned. The broad Phase C failure is not a
basic format or representation impossibility.

The no-go diagnostic weights were deleted after preserving metrics and
predictions.

### Focusedmove1

Purpose: test whether a narrower, move-quality focused continuation generalizes
better than broad Phase C.

Run root:

```text
/workspace/chess_sft_checkpoints/phase-c-focusedmove1-from-contractfix-h100-fa3-20260705T092059Z
```

Data root:

```text
/workspace/chess_sft_data/phase-c-focusedmove1-20260705T092059Z
```

Filtered curriculum:

| source | rows |
|---|---:|
| tier1 filler | 512 |
| tier2 filler | 512 |
| tier4 filler | 512 |
| tier5 filler | 512 |
| `3.1_available_captures` | 4000 |
| `3.2_threats` | 4000 |
| `3.4_tactical_patterns` | 4000 |
| `3.5_hanging_pieces` | 4000 |
| `6.3_endgame_best_move` | 2000 |
| `7.1_best_move_selection` | 2000 |
| `7.2_puzzle_solving` | 2000 |
| `7.3_move_consequence` | 2000 |

Training:

- Start checkpoint: `contractfix` best.
- Steps: 600.
- LR: `2e-6`.
- FA3, packing auto.
- Effective train rows after Phase C tier weighting/task upsampling: 75,601.
- Extra task upsampling:
  - `7.1_best_move_selection=2`
  - `7.2_puzzle_solving=2`
  - `3.4_tactical_patterns=2`
  - `3.5_hanging_pieces=2`

Training metrics:

```text
train_loss: 0.06211446523666382
num_input_tokens_seen: 38,014,064
train_runtime: 738.7217s
train_tokens_per_second: 51,459.2
```

Original planning benchmark eval, 500 examples, no ACPL:

| metric | contractfix pre | focusedmove1 post |
|---|---:|---:|
| planning overall | 15.35% | 13.89% |
| best_move | 6.29% | 5.71% |
| puzzle_solve | 14.29% | 12.50% |
| candidate_ratings | 72.78% | 70.56% |
| best_line_trace | 0.00% | 0.00% |
| step_verification | 100.00% | 94.12% |
| format_compliance | 99.58% | 98.51% |
| legal_move_rate | 64.76% | 56.26% |
| missing_move_tag_count | 0 | 0 |

Decision: no-go. The run improved train loss but worsened original planning
generalization and legality. Its weights were deleted after preserving metrics
and predictions.

Current interpretation:

- Contractfix remains the useful protocol checkpoint.
- Repair2 remains the stronger raw Phase B generator checkpoint.
- The model can memorize `7.1`/`7.2`, but supervised continuation on the current
  generated traces does not yet improve held-out move choice.
- The next change should target data quality or training objective, not another
  broader SFT continuation.

Next useful experiment:

- Build a verifier-gated/relabelled `7.1`/`7.2` set that removes noisy traces
  and trains a short answer contract, or train a move-only head/adapter-style
  diagnostic on `<move>` targets before reintroducing traces.
- Re-evaluate on original planning with exact move, legal move rate, ACPL/WPD,
  and trace referenced-move accuracy before attempting R1/R2/R3 generation.

## Move-Only Trace-Ablation Runs

The next experiment removed noisy trace prose from the supervised `7.1`/`7.2`
targets and kept only a short fixed answer:

```text
<think>Choose the final legal move.</think>
<move>{target_move}</move>
```

This tests whether Phase C is being blocked by bad trace supervision rather
than by the move target itself.

### Moveonly1

Run root:

```text
/workspace/chess_sft_checkpoints/phase-c-moveonly1-from-contractfix-h100-fa3-20260705T102350Z
```

Data root:

```text
/workspace/chess_sft_data/phase-c-moveonly1-20260705T102350Z
```

Filtered curriculum:

| source | rows |
|---|---:|
| tier1 filler | 512 |
| tier2 filler | 512 |
| tier3 filler | 512 |
| tier4 filler | 512 |
| tier5 filler | 512 |
| tier6 filler | 512 |
| `7.1_best_move_selection` move-only | 2000 |
| `7.2_puzzle_solving` move-only | 2000 |

Training:

- Start checkpoint: `contractfix` best.
- Steps: 400.
- LR: `2e-6`.
- Extra task upsampling:
  - `7.1_best_move_selection=3`
  - `7.2_puzzle_solving=3`

Training metrics:

```text
train_loss: 0.04190810641157441
num_input_tokens_seen: 24,294,416
train_runtime: 494.3127s
train_tokens_per_second: 49,147.9
```

Original planning benchmark, 500 examples:

| metric | contractfix | moveonly1 |
|---|---:|---:|
| planning overall | 15.35% | 22.51% |
| best_move | 6.29% | 8.57% |
| puzzle_solve | 14.29% | 25.36% |
| candidate_ratings | 72.78% | 79.44% |
| best_line_trace | 0.00% | 0.00% |
| step_verification | 100.00% | 100.00% |
| format_compliance | 99.58% | 98.73% |
| legal_move_rate | 64.76% | 85.77% |
| ACPL, depth 12 | 1539.5 cp | 1600.8 cp |
| WPD, depth 12 | 0.477 | 0.364 |

Decision: useful signal, but not the best checkpoint. Exact and legality
improved, but ACPL worsened. Weights were deleted after preserving metrics.

### Moveonly2

Run root:

```text
/workspace/chess_sft_checkpoints/phase-c-moveonly2-short150-from-contractfix-h100-fa3-20260705T112802Z
```

Data root:

```text
/workspace/chess_sft_data/phase-c-moveonly1-20260705T102350Z
```

Training:

- Same move-only data as moveonly1.
- Start checkpoint: `contractfix` best.
- Steps: 150.
- LR: `2e-6`.
- Same `7.1`/`7.2` task upsampling.

Training metrics:

```text
train_loss: 0.0979679787158966
num_input_tokens_seen: 9,153,544
train_runtime: 190.8451s
train_tokens_per_second: 47,963.2
```

Original planning benchmark, 500 examples with ACPL depth 12:

| metric | contractfix | moveonly2 |
|---|---:|---:|
| planning overall | 15.35% | 24.83% |
| best_move | 6.29% | 8.00% |
| puzzle_solve | 14.29% | 30.00% |
| candidate_ratings | 72.78% | 76.11% |
| best_line_trace | 0.00% | 0.00% |
| step_verification | 100.00% | 100.00% |
| format_compliance | 99.58% | 99.36% |
| legal_move_rate | 64.76% | 85.99% |
| ACPL, depth 12 | 1539.5 cp | 1447.8 cp |
| WPD, depth 12 | 0.477 | 0.346 |

Broader Phase C no-ACPL eval, 500 examples per split:

| split | contractfix | moveonly2 |
|---|---:|---:|
| perception | 99.2% | 98.8% |
| rules | 85.1% | 80.4% |
| tactics | 46.1% | 44.5% |
| evaluation | 84.2% | 83.2% |
| openings | 54.0% | 51.8% |
| endgames | 71.4% | 69.6% |
| planning | 15.5% | 24.6% |

Decision: keep. This is the best Phase C checkpoint from the July 5 sequence.
It improves planning exactness, legality, and ACPL, with manageable but real
regressions on non-planning replay.

Current best Phase C checkpoint:

```text
/workspace/chess_sft_checkpoints/phase-c-moveonly2-short150-from-contractfix-h100-fa3-20260705T112802Z/phase_c/best
```

The trainer checkpoint was deleted to save disk; `phase_c/best` remains.

### Moveonly3

Purpose: preserve non-planning behavior by adding broader Phase B replay while
keeping move-only `7.1`/`7.2`.

Run root:

```text
/workspace/chess_sft_checkpoints/phase-c-moveonly3-replay200-from-contractfix-h100-fa3-20260705T114043Z
```

Data root:

```text
/workspace/chess_sft_data/phase-c-moveonly3-replay-20260705T114043Z
```

Added replay:

- 4000 rows each from tiers 1 and 2.
- 4000 rows each from `3.1`, `3.2`, `3.4`, `3.5`.
- 4000 rows each from tiers 4 and 5.
- 3000 rows each from `6.1`, `6.2`, `6.3`, `6.4`.
- 2000 move-only rows each from `7.1` and `7.2`.

Training:

- Steps: 200.
- LR: `2e-6`.
- Train rows after mixing: capped at 100,000.

Results:

| metric | moveonly2 | moveonly3 |
|---|---:|---:|
| planning overall | 24.83% | 23.08% |
| best_move | 8.00% | 9.14% |
| puzzle_solve | 30.00% | 26.07% |
| legal_move_rate | 85.99% | 87.69% |
| ACPL, depth 12 | 1447.8 cp | 1811.7 cp |
| WPD, depth 12 | 0.346 | 0.369 |
| tactics overall | 44.5% | 45.9% |
| rules overall | 80.4% | 79.6% |

Decision: no-go. Replay helped tactics a little but hurt planning and ACPL too
much. Weights were deleted after preserving metrics.

Current interpretation after move-only ablations:

- The original trace text is a major source of harmful supervision.
- Short move-only targets give a real planning gain.
- Too much move-only training overfits; 150 steps beat 400 steps.
- Broad replay in this form does not preserve enough non-planning skill to be
  worth the planning/ACPL hit.

Next step:

- Use `moveonly2` as the current Phase C best.
- Build a proper training mode for fixed-grammar move-first data instead of
  ad hoc filtered roots.
- Add a replay-balanced move-only curriculum that explicitly controls task
  inclusion rather than relying on Phase C's hard-wired tier mix.
- Only reintroduce traces after the verifier can reject bad referenced moves and
  unfaithful PV/state/eval claims.

## Supported Move-Only Trainer Path

Implemented first-class data controls so move-only Phase C probes can run from
the canonical data root instead of copying filtered JSONL roots:

- `--task-include TASK`
- `--task-exclude TASK`
- `--move-only-task TASK`

The loader now applies these transforms before sanitized-cache writing, and the
cache key includes a transform digest. The mixer, phase dry-run, schedule dry-run,
`chess-llm-train`, and `chess-llm-run-curriculum` all forward the same transform
config.

Local verification:

```text
python -m pytest tests/test_training_data_loader.py tests/test_training_data_mixer.py tests/test_training_entrypoints.py -q
90 passed
```

H100 dry-run root:

```text
/workspace/chess_sft_checkpoints/phase-c-supported-moveonly-dryrun-20260705T143741Z
```

Dry-run result from canonical data:

```text
tier_7 summary: 60000 examples before eval split
train split: 58680 examples
eval split: 88 examples
packing: True, padding_free=True, attn=flash_attention_3
estimated optimizer steps: 150
```

Important launch note: plain `python -m chess_llm.training.train` on the 2x H100
node wraps the fp32-loaded model with `DataParallel`; FA3 then receives fp32
hidden states and fails with:

```text
FlashAttention only supports fp16, bf16, and fp8_e4m3 data type
```

The documented H100 path is DDP via `torchrun`; with DDP, bf16 autocast reaches
FA3 correctly while keeping fp32 master weights.

Supported DDP probe root:

```text
/workspace/chess_sft_checkpoints/phase-c-supported-moveonly-ddp-probe-20260705T144110Z
```

Training command:

```bash
cd /workspace/chess_sft_sdpo
source .venv-h100/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node=2 .venv-h100/bin/chess-llm-train \
  --phase c \
  --data-root /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/output \
  --output-root /workspace/chess_sft_checkpoints/phase-c-supported-moveonly-ddp-probe-20260705T144110Z \
  --benchmark-dir /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/benchmark \
  --task-include 7.1_best_move_selection \
  --task-include 7.2_puzzle_solving \
  --move-only-task 7.1_best_move_selection \
  --move-only-task 7.2_puzzle_solving \
  --task-upsample 7.1_best_move_selection=3 \
  --task-upsample 7.2_puzzle_solving=3 \
  --max-train-examples 70000 \
  --max-steps 150 \
  --learning-rate 2e-6 \
  --attn-implementation flash_attention_3 \
  --packing auto \
  --max-length 1024 \
  --skip-trainer-eval \
  --skip-eval \
  --no-wandb
```

Training metrics:

```text
train_loss: 0.1010144321123759
num_input_tokens_seen: 9,044,176
train_runtime: 181.913s
train_tokens_per_second: 49,717.0
```

Saved checkpoint:

```text
/workspace/chess_sft_checkpoints/phase-c-supported-moveonly-ddp-probe-20260705T144110Z/phase_c/best
```

Bounded sanity eval:

```bash
python -m chess_llm.training.train \
  --phase c \
  --data-root /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/output \
  --output-root /workspace/chess_sft_checkpoints/phase-c-supported-moveonly-ddp-probe-20260705T144110Z \
  --benchmark-dir /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/benchmark \
  --eval-only \
  --max-benchmark-examples-per-split 100 \
  --eval-acpl-depth 12 \
  --attn-implementation flash_attention_3 \
  --eval-batch-size 16 \
  --eval-max-new-tokens 256 \
  --no-wandb
```

Bounded eval snapshot, 100 examples per split:

| split | overall |
|---|---:|
| perception | 98.0% |
| rules | 77.5% |
| tactics | 43.7% |
| evaluation | 84.0% |
| openings | 61.7% |
| endgames | 72.0% |
| planning | 28.3% |

Planning details:

| metric | value |
|---|---:|
| best_move | 21.2% |
| puzzle_solve pass@1 | 28.8% |
| puzzle_solve pass@8 | 49.2% |
| legal_move_rate | 90.5% |
| WPD, depth 12 | 42.0% |
| ACPL, depth 12 | 1845.2 cp |

Interpretation:

- The supported canonical-data path reproduces the move-only training shape and
  keeps FA3 throughput around 50k tokens/sec on 2x H100.
- The bounded eval is not directly comparable to the earlier 500-example
  moveonly2 table, but it is a useful sanity check: planning improves strongly
  on exact move tasks, while legal-move rate and ACPL are still far from Phase C
  exit quality.
- Next real run should use this supported path, DDP launch, and a replay mix
  rather than a hand-built data root.

## Eval Speed Probe

Question: are we using CPU/GPU efficiently during eval?

Findings from the 2x H100 node:

- Transformers eval with both H100s visible uses `device_map=auto`. For this
  small 0.8B model, that shards weights across both GPUs but does not keep both
  GPUs busy. GPU0 does most generation work and GPU1 stays lightly utilized.
- A single visible H100 is faster because the evaluator takes the direct CUDA
  load path instead of cross-GPU `device_map=auto`.
- `--eval-batch-size 16` is too low. For Phase C `pass@8`, the sampled pass
  requests 7 return sequences and the effective prompt batch becomes
  `batch_size // 7`; batch 16 therefore runs sampled prompts two at a time.
- Batch 64 is the best tested setting. Batch 128 fit easily in memory but was
  slower, likely from less balanced long-generation batches.
- Stockfish diagnostics are not CPU-saturated: current eval opens one Stockfish
  process with default `Threads=1` and computes ACPL/WPD sequentially. In the
  bounded 100-example planning check this costs about 17 seconds, so generation
  is still the larger bottleneck; parallel Stockfish is a later optimization.
- vLLM is not installed in `.venv-h100`, so the current backend is Transformers.

Probe results, planning-only 100 examples, `pass@8`, no ACPL:

| setting | wall time | sampled effective batch | max GPU memory |
|---|---:|---:|---:|
| both GPUs visible, batch 16 | 44s | 2 | ~2.1 GB |
| both GPUs visible, batch 64 | 23s | 9 | ~3.8 GB |
| both GPUs visible, batch 128 | 39s | 18 | ~5.3 GB |
| `CUDA_VISIBLE_DEVICES=0`, batch 64 | 19s | 9 | ~4.7 GB |

Bounded all-split Phase C eval, 100 examples per split, `pass@8`, batch 64,
single visible H100:

| mode | wall time |
|---|---:|
| no ACPL | 58s |
| ACPL/WPD depth 12 | 75s |

Recommended eval command shape for the next run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m chess_llm.training.train \
  --phase c \
  --data-root /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/output \
  --output-root <run-root> \
  --benchmark-dir /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/benchmark \
  --eval-only \
  --max-benchmark-examples-per-split 100 \
  --eval-batch-size 64 \
  --eval-max-new-tokens 256 \
  --eval-acpl-depth 12 \
  --attn-implementation flash_attention_3 \
  --no-wandb
```

Operational recommendation: keep training with `torchrun --nproc_per_node=2`
and `--skip-eval`, then run eval as a separate single-GPU process with
`CUDA_VISIBLE_DEVICES=0 --eval-batch-size 64`. This avoids post-training eval
inside a DDP launch and leaves GPU1 free for monitoring or a separate process.

## Big Curriculum Phase C Run

Purpose: run the larger curriculum-driven Phase C pass from the current best
`moveonly2` checkpoint, with replay across the full curriculum and stronger
planning/candidate/verifier coverage.

Original run root:

```text
/workspace/chess_sft_checkpoints/phase-c-bigcurriculum1-from-moveonly2-h100-fa3-20260705T152121Z
```

Recovery run root:

```text
/root/chess_sft_checkpoints/phase-c-bigcurriculum1-stage3-resume-from-stage2-20260705T164055Z
```

Final model:

```text
/root/chess_sft_checkpoints/phase-c-bigcurriculum1-stage3-resume-from-stage2-20260705T164055Z/final_best
```

Local result artifacts:

```text
artifacts/evals/phase-c-bigcurriculum1-stage3-resume-20260705/
```

Copied artifacts:

- `final_eval_predictions.results.json`
- `final_eval_predictions.eval_run.json`
- `final_eval_predictions.analysis.json`
- `final_eval_predictions.multipv.sqlite`
- `run_bigcurriculum_recovery.log`

The raw `final_eval.log` exists on the H100 run root, but the direct SSH/SCP
route dropped while mirroring it locally. The structured result JSON is present
locally and is the source of the metrics below.

Training shape:

- Start checkpoint: `moveonly2`
  (`/workspace/chess_sft_checkpoints/phase-c-moveonly2-short150-from-contractfix-h100-fa3-20260705T112802Z/phase_c/best`).
- Stage 1 replay warmup: 800 steps, 320k max examples, LR `8e-7`.
- Stage 2 planning push: 900 steps, 360k max examples, LR `7e-7`.
- Stage 3 replay cooldown: 700 steps, 400k max examples, LR `5e-7`.
- `torchrun --nproc_per_node=2`, FA3, packing auto, max length 1024.
- `7.10_best_line_trace` excluded from training.
- `7.1_best_move_selection` and `7.2_puzzle_solving` trained as move-only
  tasks.
- Eval: 500 examples per split, pass@8, batch 64, max new tokens 256,
  Stockfish depth 12 ACPL/WPD.

Operational notes:

- Stage 1 completed.
- Stage 2 completed all 900 training steps at about 51k train tokens/sec.
- The original run failed while saving `stage2/phase_c/best` because
  `/workspace` was full:

```text
SafetensorError: Error while serializing: I/O error: No space left on device
```

- `stage2/phase_c/checkpoint-900` was complete and had `model.safetensors`.
- Stage 3 was resumed from that checkpoint and written under `/root`, which had
  sufficient free space.
- Stage 3 completed, `final_best` was saved, and final eval completed.

Final all-split eval:

| split | overall |
|---|---:|
| perception | 99.4% |
| rules | 81.0% |
| tactics | 47.2% |
| evaluation | 83.6% |
| openings | 52.9% |
| endgames | 70.4% |
| planning | 22.4% |

Planning details:

| metric | value |
|---|---:|
| best_move | 12.0% |
| puzzle_solve pass@1 | 23.2% |
| puzzle_solve pass@8 | 34.3% |
| candidate_ratings | 76.7% |
| step_verification | 100.0% |
| best_line_trace | 0.0% |
| format_compliance | 99.2% |
| legal_move_rate | 86.8% |
| WPD, depth 12 | 45.7% |
| ACPL, depth 12 | 1642.9 cp |

Comparison with `moveonly2`:

| metric | moveonly2 | big curriculum |
|---|---:|---:|
| planning overall | 24.83% | 22.44% |
| best_move | 8.00% | 12.00% |
| puzzle_solve pass@1 | 30.00% | 23.21% |
| candidate_ratings | 76.11% | 76.67% |
| legal_move_rate | 85.99% | 86.84% |
| WPD, depth 12 | 34.6% | 45.7% |
| ACPL, depth 12 | 1447.8 cp | 1642.9 cp |

Decision: no-go as the new planning-quality champion. The run improves
best-move exactness, legal move rate, candidate ratings, and broad non-planning
retention, but it hurts planning overall, puzzle solve, WPD, and ACPL.
`moveonly2` remains the planning-quality reference checkpoint. Keep this run as
a useful replay/curriculum data point, not the checkpoint to promote.
