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

### Current Critique

The project is now closer to a real package than a pile of scripts, but the
highest risk is still scientific, not mechanical. We can generate and train,
but we have not yet proven that the current curriculum produces robust chess
state tracking at scale. The next run should be sized to answer that question
directly before spending a full multi-day budget.

Main risks:

- The model may still overfit textual answer shapes without learning stable
  board coordinates.
- Multiple epochs over small generated data would waste time compared with
  scaling unique examples.
- Sidecar eval is ready enough to test, but full async checkpoint polling and
  W&B comparison still need a longer rehearsal.
- Fast kernels are useful only if tokens/sec improves on the actual workload;
  SDPA remains the baseline to beat.

### Recommended Next Experiments

1. Generate a 5K-per-task Phase A rehearsal with the current FEN mechanics
   tasks and targeted task upsampling.
2. Train one approximate pass, skip trainer eval, and run vLLM sidecar eval on
   the second GPU every saved checkpoint.
3. Compare tokens/sec, eval quality, and failure samples against the tiny
   shakedown.
4. If square/FEN mapping improves, generate a 25K-per-task run and repeat.
5. Before the full 1.73M-row Phase A target, add or expose a direct
   one-epoch/one-pass training override so the command does not rely on
   hand-computed `--max-steps`.
