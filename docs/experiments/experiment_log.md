# Experiment Log

This is the committed journal for training, eval, kernel, and curriculum
experiments. Keep secrets out of this file. Put exact artifact paths, W&B run
names, command changes, and conclusions here after each meaningful run.

## 2026-06-30 - Phase A Systems Shakedown And Real-Run Prep

### Goal

Prepare the first real SFT run for a small Qwen model that can read chess
positions, apply legal moves, and produce UCI/FEN answers reliably enough to
support later reasoning and self-distillation work.

The working curriculum idea is:

1. Teach board literacy and rules with supervised generated data.
2. Add chess reasoning and planning tasks once state mechanics are reliable.
3. Use eval failures and judged rollouts to generate targeted refresh data.
4. Move toward self-distillation and preference/SDPO-style improvement after
   the model can play legal chess consistently.

### Data Generation State

The package settings currently define 38 SFT tasks and 2.48M target examples
across all tiers. Phase A is tiers 1-2 and currently totals 1.73M target rows
before any task upsampling.

Recent Phase A changes focused on explicit FEN and square mechanics because
early eval showed that the model learned answer format faster than it learned
UCI/FEN square mapping.

- Added or emphasized `1.6_square_lookup`, `1.7_rank_lookup`,
  `1.8_move_square_edits`, `1.9_fen_assembly`,
  `1.10_fen_row_application`, `1.11_square_coordinates`,
  `1.12_fen_rank_expansion`, `1.13_fen_rank_cell_edit`, and
  `1.14_fen_board_edit`.
- Simplified state-tracking targets so Task 1.5 returns only `Result FEN`.
- Moved square/rank edit traces into dedicated mechanics tasks rather than
  duplicating them inside every state-tracking answer.
- Kept the real-run hypothesis that targeted upsampling of state tracking,
  FEN assembly, and row application should help square mapping.

Current Phase A task targets:

| Tier | Tasks | Target Rows |
| --- | ---: | ---: |
| Tier 1 perception/state | 14 | 1,320,000 |
| Tier 2 rules | 6 | 410,000 |
| Phase A total | 20 | 1,730,000 |

### Small Phase A Shakedown

This was a systems shakedown, not a meaningful capability run. Data was capped
at 100 rows per selected task for tiers 1-2.

| Item | Value |
| --- | --- |
| WSL distro | `Ubuntu-24.04-CUDA` |
| Git commit before this docs pass | `bd950a4 Prepare chess SFT package for real runs` |
| WSL repo | `/home/amazi/code/chess_sft_sdpo` |
| Main venv | `/home/amazi/code/chess_sft_sdpo/.venv` |
| Data root | `/home/amazi/chess_sft_data/phase_a_atomic_20260630` |
| Checkpoint root | `/home/amazi/chess_sft_checkpoints/phase_a` |
| Best checkpoint | `/home/amazi/chess_sft_checkpoints/phase_a/best` |
| Training rows | 2,000 |
| Eval/benchmark rows | 400 |
| Epochs used | 3 |
| Train runtime | 222.2873 seconds |
| Train loss | 0.23855089426933604 |
| Input tokens seen | 2,788,832 |
| Tokens/sec | 12,546.07 |
| Steps/sec | 1.201 |
| Samples/sec | 38.315 |

Eval artifacts:

- `/home/amazi/chess_sft_checkpoints/phase_a/eval_predictions.results.json`
- `/home/amazi/chess_sft_checkpoints/phase_a/eval_predictions.analysis.json`

Observed metrics:

| Split/Metric | Value |
| --- | ---: |
| Perception overall | 0.18 |
| Board print | 0.375 |
| Board to FEN | 0.0 |
| FEN assembly | 0.0 |
| State tracking | 0.0 |
| Square lookup | 0.142857 |
| Rules overall | 0.356349 |
| Legal moves | 0.189206 |
| Legality check | 0.357143 |
| Special rules | 0.7143 |
| Check detection | 0.0 |

Conclusion: the plumbing works, soft gates work, and W&B/eval artifacts are
usable. Capability is still poor because the run was tiny. The main diagnostic
signal is that answer format was learned before reliable square/FEN mapping.

### Training And Eval Loop Decisions

- Use W&B for real runs by default. Offline W&B should fail fast unless the run
  is an intentional smoke or rehearsal.
- Skip in-trainer eval during real Phase A training. Run benchmark eval as a
  sidecar or after checkpoints are saved.
- Keep phase gates soft while improving the model. Hard gates are useful later
  for release criteria, but they should not block iteration now.
- Prefer one pass over more generated data rather than several epochs over a
  small generated set. The current phase config still says 3 epochs, so the
  launch path needs an explicit step cap or a future epoch override before the
  first full run.

Approximate one-pass sizing from the shakedown token rate and current upsample
idea (`1.5`, `1.9`, and `1.10` at 4x):

| Data Scale | Raw Rows | Effective Rows | Approx Tokens | Approx Runtime |
| --- | ---: | ---: | ---: | ---: |
| 5K per Phase A task | 100K | 145K | 46.5M | about 1 hour |
| 25K per Phase A task | 500K | 725K | 232M | about 5 hours |
| Full Phase A targets | 1.73M | 2.63M | 840M | about 18-20 hours |

### Kernel And Attention Findings

Local training stack:

- PyTorch `2.12.1+cu130`
- Transformers `5.12.1`
- TRL `1.7.0`
- Liger installed
- FlashAttention 4 installed for explicit experiments
- `flash-linear-attention` and `causal-conv1d` installed for Qwen fast-path
  experiments

Measured/operational conclusions:

- SDPA is the safe default on the local 4090/5090 setup.
- HF FlashAttention 2 route was not faster than SDPA on the short packed Phase
  A data tested so far, so auto HF FA2 is opt-in.
- FlashAttention 4 is installed for explicit tests but should remain opt-in
  until model-load and training smoke tests are stable for the target GPU.
- Training logs now include `num_input_tokens_seen`, `train_num_tokens`, and
  `train_tokens_per_second`, so compare tokens/sec instead of examples/sec when
  packing changes.
- One source-built `causal-conv1d` fat binary should target Ada, Hopper, and
  Blackwell with `TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"` when the CUDA toolkit
  supports `sm_120`.

GPU mapping note:

- Without `CUDA_DEVICE_ORDER=PCI_BUS_ID`, local device `0` has mapped to the
  RTX 4090 and local device `1` to the RTX 5090.
- With `CUDA_DEVICE_ORDER=PCI_BUS_ID`, local device `0` has mapped to the RTX
  5090 and local device `1` to the RTX 4090.
- Always probe with PyTorch before a real launch.

### vLLM Sidecar Eval Findings

vLLM was isolated in a separate environment because the available vLLM build
uses a different torch stack than the training environment.

| Item | Value |
| --- | --- |
| vLLM venv | `/home/amazi/code/chess_sft_sdpo/.venv-vllm` |
| vLLM | `0.24.0` |
| Torch | `2.11.0+cu130` |
| Transformers | `5.12.1` |

Findings:

- vLLM's default V2 model runner failed on both local GPUs with
  `RuntimeError: UVA is not available`.
- `VLLM_USE_V1=0` was not a recognized fix.
- `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1`, `enforce_eager`, and lowering
  `max_model_len` did not fix the V2 path.
- Setting `VLLM_USE_V2_MODEL_RUNNER=0` worked on both GPUs.
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` is part of the working launcher path.
- `--vllm-max-model-len 2048` keeps eval memory bounded and avoids inheriting
  an oversized context limit.
- The vLLM path should be treated as a smoke-proven code path until a durable
  eval artifact is produced and kept under the checkpoint root.

Working smoke path:

```powershell
.\sft\training\run-wsl.ps1 `
  -NoSync `
  -VenvPath /home/amazi/code/chess_sft_sdpo/.venv-vllm `
  -- bash -lc 'export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn; chess-llm-evaluate --model /home/amazi/chess_sft_checkpoints/phase_a/best --benchmark-dir /home/amazi/chess_sft_data/phase_a_atomic_20260630/benchmark --output /home/amazi/chess_sft_checkpoints/phase_a/vllm_eval_predictions.jsonl --inference-backend vllm --vllm-max-model-len 2048 --soft-gate'
```

The first pass compiles kernels; repeated runs should be faster after caches are
warm.

### Phase A 25k Rehearsal Preflight - 2026-06-30

Data root:

- `/home/amazi/chess_sft_data/phase-a-t12-v25000-20260630`

Generation and validation:

- Generated tiers 1-2 with `--volume 25000`: 20 task files, 500,000 raw rows.
- Frozen benchmark: 4,000 examples across perception and rules.
- Validation pass: 500,000 examples, 0 validation errors.
- Contamination check: no benchmark/FEN blocklist contamination found.
- Critical task spot checks all matched `metadata.expected_answer` for 25,000
  rows each: `1.5`, `1.9`, `1.10`, `1.13`, `1.14`, `2.1`, `2.3`.
- `1.9_fen_assembly` now has lookup, square edit, rank edit, and exactly one
  final `Result FEN:` line for every checked row.
- `2.3_move_legality_check` mix: 12,552 legal and 12,448 illegal rows.

Dry-run config:

- Pre-split effective rows after task upsampling: 725,000.
- Actual train/eval split: 708,379 train rows, 10,000 eval rows.
- One-pass estimate: 22,137 optimizer steps, 665 warmup steps.
- Trainer eval disabled; step checkpoints every 1,000 for the real rehearsal.
- Periodic checkpoints are now resumable: model, optimizer, scheduler, RNG, and
  `trainer_state.json` are saved.

Warmup:

- Checkpoint root:
  `/home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-20260630-warmup-ada6ae4`
- Part 1 ran 60 optimizer steps from scratch and wrote full-state checkpoints.
- Part 2 resumed with `--resume-from-checkpoint auto` from `checkpoint-60` and
  trained through step 120.
- Training attention: `auto` selected SDPA.
- Liger applied to Qwen3 with `cross_entropy=False` and
  `fused_linear_cross_entropy=False`.
- Observed token throughput:
  - 60-step fresh segment: about 10.6k input tokens/sec.
  - 60-step resumed segment: about 20.7k input tokens/sec.

Forced FA2 smoke:

- Checkpoint root:
  `/home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-20260630-fa2-smoke-06eec45`
- Training attention was forced to the pinned HF
  `kernels-community/flash-attn2@bcc70a66fbe0445d4484f56167d706134d53633d`
  implementation on the RTX 5090.
- TRL packing was enabled and completed quickly, but actual training was far
  slower than the SDPA baseline.
- Observed token throughput: about 2.85k input tokens/sec over 20 optimizer
  steps, with about 31.9 GB of 5090 memory used.
- Decision: keep the 25k rehearsal on `--attn-implementation auto`/SDPA.
  Re-test FA2 only with a smaller sequence/batch shape or a different kernel
  stack.

Sidecar eval:

- vLLM eval ran from `.venv-vllm` on the warmup `best/` checkpoint with
  `VLLM_USE_V2_MODEL_RUNNER=0`, `--vllm-max-model-len 4096`, and 100 examples
  per split.
- vLLM used FlashAttention v2 and wrote predictions, analysis, results, and
  eval-run metadata under the warmup checkpoint root.
- W&B eval artifact upload succeeded.
- Metrics were intentionally poor after only 120 training steps, but the
  sidecar eval path is mechanically proven on the rehearsal artifacts.

### Phase A 25k One-Pass Rehearsal - 2026-06-30

Run identity:

- Data root:
  `/home/amazi/chess_sft_data/phase-a-t12-v25000-20260630`
- Checkpoint root:
  `/home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-20260630`
- Git SHA: `a48e06c316259be3f63cc55f277ec304cf62b66a`
- W&B train run: `phase-a-t12-v25000-20260630-train`
  (`u04sgfma`)
- W&B final eval run: `phase-a-t12-v25000-20260630-vllm-sidecar-eval`
  (`id5v6692`)

Training configuration:

- `--num-train-epochs 1`
- `--skip-trainer-eval`
- `--skip-eval`
- `--no-acpl`
- Task upsampling: `1.5_state_tracking=4`, `1.9_fen_assembly=4`,
  `1.10_fen_row_application=4`
- Attention request: `auto`; selected backend: SDPA.
- Liger applied to Qwen3 with `cross_entropy=False` and
  `fused_linear_cross_entropy=False`.
- Step checkpoints every 1,000 steps with full resume state.

Training result:

- Optimizer steps: 22,137.
- Train rows after split and upsampling: 708,379.
- Eval holdout rows: 10,000.
- Input tokens seen: 228,134,416.
- Train runtime: 4:59:56.
- Train throughput: 12,676.7 input tokens/sec.
- Train loss: 0.0493.
- Final export: `phase_a/best`.
- `READY`, `train_results.json`, `all_results.json`, final predictions,
  final analysis, and checkpoint eval artifacts were all written.

Sidecar eval setup:

- vLLM ran from `/home/amazi/code/chess_sft_sdpo/.venv-vllm`.
- `VLLM_USE_V2_MODEL_RUNNER=0`.
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- `--vllm-max-model-len 4096`.
- Periodic checkpoint evals used 200 examples per split.
- Final eval used 500 examples per split.
- Eval ran on the second GPU path (`CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `CUDA_VISIBLE_DEVICES=1`, RTX 4090 on this workstation) while training ran on
  the RTX 5090.

Checkpoint eval trend:

| Step | Perception | State | FEN Assembly | FEN Row | Material | Rules | Legal Moves | Legality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 46.5% | 14.3% | 7.1% | 7.1% | 6.7% | 51.9% | 11.2% | 57.6% |
| 5,000 | 84.5% | 78.6% | 78.6% | 78.6% | 0.0% | 65.0% | 13.0% | 78.8% |
| 10,000 | 90.0% | 85.7% | 85.7% | 85.7% | 20.0% | 68.8% | 19.5% | 75.8% |
| 15,000 | 91.0% | 85.7% | 85.7% | 85.7% | 26.7% | 70.0% | 25.7% | 75.8% |
| 20,000 | 91.0% | 85.7% | 85.7% | 85.7% | 20.0% | 70.2% | 24.2% | 75.8% |

Final eval:

| Split | Metric | Score |
| --- | --- | ---: |
| Perception | overall | 92.6% |
| Perception | board_print | 100.0% |
| Perception | board_to_fen | 100.0% |
| Perception | square_lookup | 100.0% |
| Perception | fen_assembly | 85.7% |
| Perception | fen_row_application | 94.3% |
| Perception | state_tracking | 91.4% |
| Perception | material_count | 30.6% |
| Rules | overall | 68.0% |
| Rules | legal_moves | 26.1% |
| Rules | piece_legal_moves | 52.3% |
| Rules | side_piece_inventory | 87.7% |
| Rules | legality_check | 77.1% |
| Rules | check_detection | 91.6% |
| Rules | special_rules | 73.5% |

Interpretation:

- The square/FEN curriculum worked. By 10k steps, the model crossed the Phase A
  state-tracking target on the sidecar sample, and the final 500-example eval
  reached 91.4% state tracking.
- The main 1.9/1.10 simplification paid off: FEN assembly and row application
  are no longer random format imitation. Remaining FEN misses are usually
  capture-square, halfmove-clock, or one-cell placement mistakes.
- Legal move generation is still the main bottleneck. The model often
  over-generates pseudo-legal moves, misses blockers/check constraints, or
  falls back to opening-position pawn/knight templates.
- Material count is still a weak separate skill. It improved late in the run,
  but exact count remains poor even when piece-family accuracy is much higher.
- Later training after about 10k steps produced small but useful rules/material
  gains while FEN/state mostly plateaued.
- The final eval still failed the Phase A soft criteria on legal moves,
  legality check, and the floor check. That is expected for this rehearsal, but
  it argues against spending the full Phase A budget on the unchanged recipe.

### Current Critique

The project is now closer to a real package than a pile of scripts, but the
highest risk is still scientific, not mechanical. We can generate, train,
checkpoint, resume, and sidecar-evaluate. The 25k rehearsal proved that the
current curriculum can teach square/FEN mechanics at this model scale, but it
also showed that legal move enumeration and exact material accounting need more
explicit decomposition before the full Phase A budget is worth spending.

Main risks:

- The model can now read and edit boards, but legal move generation remains too
  unconstrained.
- Material count exact-match quality is much worse than piece-family accuracy,
  so the scoring target likely needs a staged count inventory curriculum.
- More examples of the same mixed recipe may help, but the trend suggests it
  will be inefficient for the weakest rules tasks.
- Multiple epochs over small generated data would still waste time compared
  with scaling unique examples.
- Fast kernels are useful only if tokens/sec improves on the actual workload;
  SDPA remains the baseline to beat for training.

### Recommended Next Experiments

1. Add targeted material-count decomposition: side inventory, per-piece counts,
   piece values, subtotal arithmetic, and final material sentence as separate
   tasks.
2. Add targeted legal-move decomposition before full enumeration: side-piece
   inventory, piece-ray blockers, attacked king filtering, pinned pieces, and
   legal-vs-pseudo-legal contrast sets.
3. Run a smaller targeted rules/material rehearsal before scaling the full
   Phase A recipe.
4. Keep the FEN/state tasks in the mix, but reduce their upsampling once they
   consistently stay above the target.
5. Use the final 25k rehearsal eval as the baseline for the next run.
