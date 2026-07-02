# Phase A Real-Run Runbook

This runbook is for the first serious Phase A SFT run: tiers 1-2, one pass over
fresh generated data, trainer eval disabled, W&B enabled, and benchmark eval
handled separately.

## Source Of Truth

- Package settings: `src/chess_llm/sft/settings.py`
- Training phases: `src/chess_llm/training/phases.py`
- Training CLI: `chess-llm-train`
- Data CLI: `chess-llm-make-data`
- Eval CLI: `chess-llm-evaluate`
- WSL launcher: `sft/training/run-wsl.ps1`
- Experiment journal: `docs/experiments/experiment_log.md`

Current target volumes:

| Scope | Tasks | Target Rows |
| --- | ---: | ---: |
| Tier 1 perception/state/material mechanics | 18 | 1,520,000 |
| Tier 2 rules/legal-move decomposition | 10 | 650,000 |
| Phase A total | 28 | 2,170,000 |
| All tiers | 38 | 2,480,000 |

## Launch Principles

- Prefer more unique generated examples over multiple epochs on small data.
- Skip trainer eval for real runs with `--skip-trainer-eval`.
- Keep gates soft while iterating. Use `--require-phase-gate` later for release
  criteria, not for the first improvement loops.
- Use W&B online for real training and direct eval. Do not run a real launch
  with `WANDB_MODE=offline`.
- Compare speed with tokens/sec, not examples/sec, because packing and output
  lengths change the number of useful tokens processed.
- Treat SDPA as the stable baseline. Use FA2/FA4 only after a model-load and
  short train smoke succeeds on the selected GPU.
- Use `--attn-implementation sdpa --packing off --max-length 1024` for the
  current rehearsal and full Phase A launch. On the 2026-07-02 controlled
  smoke with `Qwen/Qwen3.5-0.8B` and the refreshed legal/material data,
  unpacked SDPA reached about 5.1k train tokens/sec while TRL packing reached
  about 1.2k train tokens/sec on the same 100-step budget.
- The 2026-07-01 HF FA2 smoke did not complete 100 steps in 40 minutes on this
  stack. Keep FA2 disabled until a short capped smoke proves it is faster than
  the SDPA packed path.

## Latest Rehearsal Outcome

The 2026-06-30 25k-per-task rehearsal completed one pass successfully on SDPA:

- 22,137 optimizer steps.
- 228.1M input tokens.
- 4:59:56 train runtime.
- 12.7k input tokens/sec.
- Final vLLM sidecar eval: 92.6% perception overall, 68.0% rules overall.
- State tracking reached 91.4%; legal move generation remained low at 26.1%.

Decision: do not scale the unchanged 2026-06-30 recipe directly to the old
1.73M-row Phase A target. Add targeted material-count and legal-move
decomposition first, then run a smaller focused rehearsal and compare against
the 25k baseline.

Latest 2026-07-02 prelaunch smoke on the refreshed legal/material data:

- Data root: `/home/amazi/chess_sft_data/phase-a-t12-v25000-legal-material-20260702/output`
- Base model: `Qwen/Qwen3.5-0.8B`
- Dry-run train split: 903,059 examples after the current task upsampling.
- Packing off, 100 steps: 1,120,160 train tokens, 220.2s runtime, about
  5,087 train tokens/sec.
- Packing on, 100 steps: 3,203,624 train tokens, 2,666.8s runtime, about
  1,201 train tokens/sec.
- vLLM sidecar on the 4090 can load the text-only SFT checkpoint through the
  automatic `best/vllm_qwen35_wrapper` export and uses FlashAttention v2 plus
  Qwen GDN/conv kernels.

Use the unpacked SDPA path for the next serious rehearsal and full Phase A
target unless a newer controlled smoke beats it on train tokens/sec.

## Preflight Checklist

From the Windows repo root:

```powershell
.\sft\training\run-wsl.ps1 -Distro Ubuntu-24.04-CUDA -- python -c "import torch; print(torch.__version__); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
.\sft\training\run-wsl.ps1 -Distro Ubuntu-24.04-CUDA -- chess-llm-preflight --help
.\sft\training\run-wsl.ps1 -Distro Ubuntu-24.04-CUDA -- chess-llm-train --phase a --dry-run
```

Confirm before a real run:

- `.env` is populated with Hugging Face and W&B credentials.
- `WANDB_MODE` is unset or set to `online`.
- `WANDB_GIT_COMMIT` is set to the pushed Windows git SHA before train/eval
  because the WSL source tree may be rsynced without `.git`.
- WSL repo path is `/home/amazi/code/chess_sft_sdpo` unless intentionally
  changed.
- Training venv is `/home/amazi/code/chess_sft_sdpo/.venv`.
- vLLM eval venv is `/home/amazi/code/chess_sft_sdpo/.venv-vllm`.
- PyTorch GPU order was checked in the same environment used for launch.
- Data and checkpoint roots point to Linux filesystem paths, not `/mnt/c`, for
  better I/O.
- `-WslDataRoot` is the `CHESS_SFT_OUTPUT` root containing `output/` and
  `benchmark/`; it is not the `output/` directory itself.

## Data Scale Ladder

Use a ladder instead of jumping straight to the full target:

| Step | Generation Mode | Purpose |
| --- | --- | --- |
| Smoke | `--volume 100` | Validate sources, writes, eval split, and training path |
| Rehearsal | `--volume 5000` | Test real loop behavior in about an hour |
| Medium | `--volume 25000` | Check scaling and capability trend before full cost |
| Full Phase A | default volumes | Real Phase A target after the loop is proven |

`--volume` is a uniform per-selected-task override for smoke and rehearsals.
Omit it to use the per-task targets from `src/chess_llm/sft/settings.py`.

## Generate Data

Example serious rehearsal generation:

```powershell
.\sft\training\run-wsl.ps1 `
  -WslRepoPath /home/amazi/code/chess_sft_sdpo `
  -VenvPath /home/amazi/code/chess_sft_sdpo/.venv `
  -WslDataRoot /home/amazi/chess_sft_data/phase_a_25k `
  -- chess-llm-make-data --tier 1 2 --volume 25000 --source-readiness-report /home/amazi/chess_sft_data/phase_a_25k/readiness.json
```

After generation:

```powershell
.\sft\training\run-wsl.ps1 `
  -NoSync `
  -WslDataRoot /home/amazi/chess_sft_data/phase_a_25k `
  -- chess-llm-validate-outputs --output-dir /home/amazi/chess_sft_data/phase_a_25k/output --blocklist /home/amazi/chess_sft_data/phase_a_25k/eval_splits/blocklist.txt --expected-volume 25000 --tier 1 2
```

For the full run, omit `--volume` and use a fresh output root.

## Training Command Shape

The current phase config still defines 3 epochs for Phase A. For a real
generated-data run, use `--num-train-epochs 1` so the trainer makes one pass
over the mixed and task-upsampled dataset. Keep `--max-steps` for bounded
warmups only.

Current defaults are `per_device_train_batch_size=4` and
`gradient_accumulation_steps=8`. As a rough guide with one GPU and the current
`1.5`, `1.9`, `1.10` upsampling idea, dry-run row counts map directly to the
unpacked training plan. The 2026-07-02 legal/material dry-run reported 925,000
pre-split effective rows and 903,059 train rows, or about 28,221 optimizer
steps at effective batch size 32. Estimate wall time from capped smokes on the
exact data root, with tokens/sec as the comparison metric.

Recommended current task emphasis:

- `1.5_state_tracking=4`
- `1.9_fen_assembly=4`
- `1.10_fen_row_application=4`

Command template:

```powershell
$env:WANDB_GIT_COMMIT = (git rev-parse HEAD).Trim()

.\sft\training\run-wsl.ps1 `
  -WslRepoPath /home/amazi/code/chess_sft_sdpo `
  -VenvPath /home/amazi/code/chess_sft_sdpo/.venv `
  -WslDataRoot /home/amazi/chess_sft_data/phase_a_25k `
  -WslCheckpointRoot /home/amazi/chess_sft_checkpoints/phase_a_25k `
  -WslHfCache /home/amazi/.cache/huggingface `
  -WslWandbDir /home/amazi/chess_sft_wandb `
  -- chess-llm-train --phase a `
  --attn-implementation sdpa `
  --packing off `
  --max-length 1024 `
  --num-train-epochs 1 `
  --task-upsample 1.5_state_tracking=4 `
  --task-upsample 1.9_fen_assembly=4 `
  --task-upsample 1.10_fen_row_application=4 `
  --skip-trainer-eval `
  --trainer-save-steps 1000 `
  --skip-eval `
  --wandb-project chess-sft `
  --wandb-group phase-a `
  --run-name phase-a-25k-one-pass
```

With `--skip-trainer-eval`, `best/` is the final exported model, not the
best-by-eval-loss checkpoint. `--trainer-save-steps` still creates periodic
recovery checkpoints with Trainer state. To resume an interrupted run from the
latest saved step checkpoint, relaunch the same training command with:

```text
--resume-from-checkpoint auto
```

You can also pass an explicit `checkpoint-N` directory.

For a bounded smoke, also add:

```text
--max-train-examples 8192 --max-steps 100
```

Keep `--packing auto` or `--packing on` only for smoke comparisons. In the
current Qwen3.5 + TRL stack, packing was slower on the controlled 100-step
comparison, so the production path should pass `--packing off` explicitly.

## vLLM Sidecar Eval

Keep vLLM in its separate venv:

```text
/home/amazi/code/chess_sft_sdpo/.venv-vllm
```

On the current Ubuntu 24 WSL setup, vLLM's default V2 runner hit
`RuntimeError: UVA is not available`. The working path is to disable the V2
model runner:

Fine-tuned Qwen3.5 SFT checkpoints save as text-only `qwen3_5_text`
checkpoints, while vLLM loads the base model through the wrapper
`Qwen3_5ForConditionalGeneration` architecture. `chess-llm-evaluate` now
auto-exports local Qwen3.5 text checkpoints to `best/vllm_qwen35_wrapper` for
vLLM by hardlinking the SFT language weights and base visual/config weights.
You can also create that export explicitly:

```powershell
.\sft\training\run-wsl.ps1 `
  -VenvPath /home/amazi/code/chess_sft_sdpo/.venv-vllm `
  -- chess-llm-export-vllm --checkpoint /home/amazi/chess_sft_checkpoints/phase_a_25k/best
```

```powershell
$env:WANDB_GIT_COMMIT = (git rev-parse HEAD).Trim()
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:VLLM_USE_V2_MODEL_RUNNER = '0'
$env:VLLM_WORKER_MULTIPROC_METHOD = 'spawn'

.\sft\training\run-wsl.ps1 `
  -NoSync `
  -CudaDeviceId 0 `
  -VenvPath /home/amazi/code/chess_sft_sdpo/.venv-vllm `
  -- chess-llm-evaluate --model /home/amazi/chess_sft_checkpoints/phase_a_25k/best --benchmark-dir /home/amazi/chess_sft_data/phase_a_25k/benchmark --output /home/amazi/chess_sft_checkpoints/phase_a_25k/vllm_eval_predictions.jsonl --phase a --inference-backend vllm --vllm-max-model-len 4096 --max-new-tokens 384 --soft-gate --no-acpl --wandb-project chess-sft --wandb-group phase-a-eval --wandb-run-name phase-a-25k-vllm-eval
```

Use the second GPU for sidecar eval. In the current launch-wrapper probe,
`-CudaDeviceId 1` exposes the RTX 5090 to PyTorch for training and
`-CudaDeviceId 0` exposes the RTX 4090 for eval. Re-probe before launch because
CUDA device ordering is environment-sensitive.

Sidecar eval can poll saved checkpoints in a separate terminal, but the first
real rehearsal should keep this simple: evaluate each saved checkpoint manually
or with a small script after the checkpoint directory appears.

## After The Run

Record in `docs/experiments/experiment_log.md`:

- Data root and checkpoint root.
- Git commit.
- W&B run name and group.
- Training tokens/sec and total tokens.
- Attention backend and packing state.
- Eval backend and GPU.
- Benchmark headline metrics.
- `legal_moves_by_piece` exact score plus partial diagnostics:
  `all_moves_jaccard`, `all_moves_precision`, `all_moves_recall`,
  `per_piece_group_jaccard`, `section_completeness`, `illegal_extra_count`,
  and `missing_move_count`.
- The top recurring failure modes from `eval_predictions.analysis.json`.
- Decision for the next data scale.
