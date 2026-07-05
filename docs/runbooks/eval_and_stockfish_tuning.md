# Eval And Stockfish Tuning Runbook

Use these probes before a real training run when eval speed, GPU batching, or
Stockfish throughput is uncertain. They are intentionally small and bounded:
the goal is to find the best launch shape, not to score a model deeply.

## Source Of Truth

- Eval CLI: `chess-llm-evaluate` or `python -m chess_llm.training.evaluate`
- Training eval wrapper: `python -m chess_llm.training.train --eval-only`
- Eval speed probe: `scripts/tune_eval_speed.py`
- Stockfish speed probe: `scripts/tune_stockfish_speed.py`
- Current H100 experiment note:
  `docs/experiments/runs/2026-07-05_phase_c_contractfix.md`

## Eval Probe

The eval probe sweeps visible GPU sets and generation batch sizes, captures wall
time, and samples `nvidia-smi` when available.

Example on the 2x H100 node:

```bash
cd /workspace/chess_sft_sdpo
source .venv-h100/bin/activate

python scripts/tune_eval_speed.py \
  --model /workspace/chess_sft_checkpoints/phase-c-supported-moveonly-ddp-probe-20260705T144110Z/phase_c/best \
  --benchmark-dir /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/benchmark \
  --output-dir /workspace/chess_sft_checkpoints/eval-speed-probe-$(date +%Y%m%dT%H%M%SZ) \
  --cuda-visible-devices '0;0,1' \
  --batch-sizes '16,64,128' \
  --split planning \
  --max-examples-per-split 100 \
  --pass-k 8 \
  --max-new-tokens 256 \
  --attn-implementation flash_attention_3
```

Use semicolons between CUDA masks. `0,1` is one mask; `0;0,1` means "run once
with only GPU 0 visible, then run once with GPUs 0 and 1 visible."

Outputs:

- `eval_speed_summary.json`
- `eval_speed_summary.md`
- one eval output JSONL per probe
- one log per probe
- one `nvidia-smi` CSV per probe when `nvidia-smi` exists

Interpretation:

- Compare wall time first.
- Then check max memory and average GPU utilization.
- For `pass@8`, remember that sampled generation requests 7 extra return
  sequences, so effective prompt batch is roughly `batch_size // 7`.
- If both GPUs are visible for a small model, Transformers may use
  `device_map=auto` and shard weights without improving generation throughput.
  Single-GPU eval can be faster.

Current H100 default:

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

For training runs, prefer DDP training with `--skip-eval`, then run eval
separately with one visible H100 and batch 64.

## Stockfish Probe

The Stockfish probe measures true root MultiPV analysis throughput over FENs
from benchmark JSONL. It varies the number of Stockfish processes and
`Threads` per process.

Example on the H100 node:

```bash
cd /workspace/chess_sft_sdpo
source .venv-h100/bin/activate

python scripts/tune_stockfish_speed.py \
  --benchmark-dir /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/benchmark \
  --split planning \
  --output-dir /workspace/chess_sft_checkpoints/stockfish-speed-probe-$(date +%Y%m%dT%H%M%SZ) \
  --stockfish-path /usr/games/stockfish \
  --limit 200 \
  --depth 12 \
  --multipv 5 \
  --worker-counts '1,4,8,16,32' \
  --threads-per-worker '1,2,4' \
  --hash-mb 128
```

Outputs:

- `stockfish_speed_summary.json`
- `stockfish_speed_summary.md`

Interpretation:

- Prefer the highest stable `positions_per_second`.
- Start with many workers and `threads_per_worker=1`; this usually fits our
  workload better than one engine with many threads.
- Total Stockfish hash memory is `workers * hash_mb`. Keep it small enough that
  the OS page cache and Python process are not squeezed.
- If `threads_per_worker=2` or `4` does not beat `1`, keep `1`. Extra threads
  can reduce throughput when many independent FENs are available.
- This probe measures root MultiPV only. Full WPD eval can also need post-move
  fallback analysis when the model move is outside cached top-N.

## Recommended Order Before A Run

1. Run `scripts/tune_eval_speed.py` on 100 planning examples with no ACPL.
2. Pick the fastest GPU mask and batch size.
3. Run the selected eval shape once with ACPL/WPD enabled at the target depth.
4. Run `scripts/tune_stockfish_speed.py` with the same depth and `multipv=5`.
5. If Stockfish is a visible bottleneck, use the probe result to set
   `--stockfish-workers`; ACPL cache misses are parallelized, while WPD/MultiPV
   fallback scoring still runs through the main engine path.
6. Launch training with `torchrun --nproc_per_node=2 ... --skip-eval`.
7. Run bounded eval separately first, then full eval only for promising
   checkpoints.

## Current H100 Findings

From the 2026-07-05 Phase C sequence:

- 2x H100 DDP training with FA3 reaches about 50k train tokens/sec.
- Plain non-DDP training on both H100s can route through DataParallel and fail
  FA3 dtype expectations; use `torchrun`.
- For the 0.8B model, single-H100 eval was faster than two visible H100s.
- `--eval-batch-size 64` was the best tested Phase C pass@8 setting.
- Batch 128 fit in memory but was slower.
- Bounded all-split eval at 100 examples/split took about 58s without ACPL and
  about 75s with ACPL/WPD depth 12.
- Current eval Stockfish use supports parallel ACPL cache misses via
  `--stockfish-workers`, but short bounded evals are still usually dominated by
  generation batching before Stockfish saturation.
