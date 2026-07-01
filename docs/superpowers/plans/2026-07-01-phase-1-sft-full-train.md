# Phase 1 SFT Full Train Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare, launch, monitor, and analyze the first full Phase 1 / Phase A SFT run for tiers 1-2 using one pass over freshly generated data, W&B logging, and vLLM sidecar eval.

**Architecture:** Keep data generation, training, and eval as separate artifacts with explicit Linux filesystem roots. Generate tiers 1-2 into a fresh WSL data root, train one epoch with trainer eval disabled, then evaluate checkpoints with vLLM on the second GPU. The full launch is gated by a focused rehearsal and dry-run estimates, not by hard metric gates.

**Tech Stack:** Python package `chess_llm`, `chess-llm-make-data`, `chess-llm-train`, `chess-llm-evaluate`, Ubuntu-24.04-CUDA WSL, PyTorch, TRL/SFTTrainer, W&B, vLLM, PowerShell launch wrappers.

---

## Current Phase A Target

Current default volumes in `src/chess_llm/sft/settings.py`:

- Tier 1 rows: 1,520,000
- Tier 2 rows: 650,000
- Phase A raw rows: 2,170,000
- One-pass effective rows with `1.5`, `1.9`, and `1.10` upsampled to factor 4: about 3,070,000 rows before train/eval split

Use the trainer dry-run output as the source of truth for row counts before packing. Use a capped packed smoke on the exact data root as the source of truth for optimizer step time and wall-clock estimates. The working 2026-07-01 training path is `--attn-implementation sdpa --packing on --max-length 1024`; the older `auto` path did not pack and was stopped because it projected to a multi-day run.

## Files And Artifacts

- Read: `docs/runbooks/phase_a_real_run.md`
- Read/update after runs: `docs/experiments/experiment_log.md`
- Read: `src/chess_llm/sft/settings.py`
- Read: `src/chess_llm/training/phases.py`
- Use: `sft/training/run-wsl.ps1`
- Optional helper references: `.tmp/run_phase_a_t12_generate.ps1`, `.tmp/run_phase_a_t12_train.ps1`, `.tmp/run_phase_a_t12_vllm_sidecar_eval.ps1`
- Rehearsal data root: `/home/amazi/chess_sft_data/phase-a-t12-v25000-20260701-decomp`
- Rehearsal checkpoint root: `/home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-20260701-decomp-pack1024`
- Full data root: `/home/amazi/chess_sft_data/phase-a-t12-full-20260701`
- Full checkpoint root: `/home/amazi/chess_sft_checkpoints/phase-a-t12-full-20260701-pack1024`

### Task 1: Freeze The Code State

**Files:**
- Read: `git status`
- Modify only if needed: tracked source/docs/tests already changed in the working tree

- [ ] **Step 1: Verify tests still pass**

```powershell
python -m pytest -q
```

Expected: exit code `0` with the full test suite passing.

- [ ] **Step 2: Inspect the working tree**

```powershell
git status --short
git diff --stat
```

Expected: only intentional curriculum, validation, benchmark, training, and plan changes are present.

- [ ] **Step 3: Commit the launch-ready code**

```powershell
git add src tests docs
git commit -m "feat: add training packing controls"
git push origin HEAD
```

Expected: commit and push succeed. Copy the resulting SHA with:

```powershell
git rev-parse HEAD
```

- [ ] **Step 4: Pin the SHA for W&B**

```powershell
$env:WANDB_GIT_COMMIT = (git rev-parse HEAD).Trim()
Write-Host $env:WANDB_GIT_COMMIT
```

Expected: W&B commit value matches the pushed SHA.

### Task 2: Verify WSL, Tokens, Kernels, And W&B

**Files:**
- Read: `.env`
- Read: `sft/training/run-wsl.ps1`
- Read: `src/chess_llm/training/training_args.py`

- [ ] **Step 1: Probe GPUs in the launch distro**

```powershell
.\sft\training\run-wsl.ps1 `
  -Distro Ubuntu-24.04-CUDA `
  -- python -c "import torch; print(torch.__version__); print(torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

Expected: both GPUs are visible. Record which index maps to RTX 5090 and RTX 4090.

- [ ] **Step 2: Verify package entrypoints**

```powershell
.\sft\training\run-wsl.ps1 -Distro Ubuntu-24.04-CUDA -- chess-llm-make-data --help
.\sft\training\run-wsl.ps1 -Distro Ubuntu-24.04-CUDA -- chess-llm-train --help
.\sft\training\run-wsl.ps1 -Distro Ubuntu-24.04-CUDA -- chess-llm-evaluate --help
```

Expected: each command exits `0`.

- [ ] **Step 3: Run a training dry-run against the current default environment**

```powershell
.\sft\training\run-wsl.ps1 `
  -Distro Ubuntu-24.04-CUDA `
  -- chess-llm-train --phase a --attn-implementation sdpa --packing on --max-length 1024 --num-train-epochs 1 --skip-trainer-eval --skip-eval --dry-run
```

Expected: command exits `0` and prints Phase A config. If W&B is enabled but credentials are missing, fix `.env` or `wandb login` inside WSL before continuing.

### Task 3: Generate And Validate The 25k Decomposition Rehearsal

**Files:**
- Create external artifact: `/home/amazi/chess_sft_data/phase-a-t12-v25000-20260701-decomp`
- Read/update after run: `docs/experiments/experiment_log.md`

- [ ] **Step 1: Generate tiers 1-2 at 25k rows per task**

```powershell
.\sft\training\run-wsl.ps1 `
  -WslRepoPath /home/amazi/code/chess_sft_sdpo `
  -VenvPath /home/amazi/code/chess_sft_sdpo/.venv `
  -WslDataRoot /home/amazi/chess_sft_data/phase-a-t12-v25000-20260701-decomp `
  -- chess-llm-make-data `
  --tier 1 2 `
  --volume 25000 `
  --eval-split-volume 25000 `
  --source-readiness-report /home/amazi/chess_sft_data/phase-a-t12-v25000-20260701-decomp/readiness.json
```

Expected: 28 tier task files are generated and the command exits `0`.

- [ ] **Step 2: Validate generated outputs**

```powershell
.\sft\training\run-wsl.ps1 `
  -NoSync `
  -WslDataRoot /home/amazi/chess_sft_data/phase-a-t12-v25000-20260701-decomp `
  -- chess-llm-make-data `
  --validate-only `
  --tier 1 2 `
  --volume 25000
```

Expected: validation exits `0` with completeness satisfied for tiers 1-2.

- [ ] **Step 3: Dry-run the exact rehearsal train config**

```powershell
.\sft\training\run-wsl.ps1 `
  -NoSync `
  -WslDataRoot /home/amazi/chess_sft_data/phase-a-t12-v25000-20260701-decomp `
  -WslCheckpointRoot /home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-20260701-decomp-pack1024 `
  -- chess-llm-train `
  --phase a `
  --data-root /home/amazi/chess_sft_data/phase-a-t12-v25000-20260701-decomp/output `
  --output-root /home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-20260701-decomp-pack1024 `
  --benchmark-dir /home/amazi/chess_sft_data/phase-a-t12-v25000-20260701-decomp/benchmark `
  --attn-implementation sdpa `
  --packing on `
  --max-length 1024 `
  --num-train-epochs 1 `
  --task-upsample 1.5_state_tracking=4 `
  --task-upsample 1.9_fen_assembly=4 `
  --task-upsample 1.10_fen_row_application=4 `
  --skip-trainer-eval `
  --trainer-save-steps 1000 `
  --skip-eval `
  --no-acpl `
  --wandb-project chess-sft `
  --wandb-group phase-a-t12-v25000-20260701-decomp-pack1024 `
  --run-name phase-a-t12-v25000-20260701-decomp-pack1024-dry-run `
  --dry-run
```

Expected: dry-run exits `0` and prints row counts, split sizes, upsampling counts, and estimated optimizer steps.

### Task 4: Run The Focused Rehearsal And Sidecar Eval

**Files:**
- Create external artifact: `/home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-20260701-decomp-pack1024`
- Update after run: `docs/experiments/experiment_log.md`

- [ ] **Step 1: Launch the 25k one-pass rehearsal**

```powershell
$env:WANDB_GIT_COMMIT = (git rev-parse HEAD).Trim()

.\sft\training\run-wsl.ps1 `
  -WslRepoPath /home/amazi/code/chess_sft_sdpo `
  -VenvPath /home/amazi/code/chess_sft_sdpo/.venv `
  -WslDataRoot /home/amazi/chess_sft_data/phase-a-t12-v25000-20260701-decomp `
  -WslCheckpointRoot /home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-20260701-decomp-pack1024 `
  -WslHfCache /home/amazi/.cache/huggingface `
  -WslWandbDir /home/amazi/chess_sft_wandb `
  -- chess-llm-train `
  --phase a `
  --data-root /home/amazi/chess_sft_data/phase-a-t12-v25000-20260701-decomp/output `
  --output-root /home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-20260701-decomp-pack1024 `
  --benchmark-dir /home/amazi/chess_sft_data/phase-a-t12-v25000-20260701-decomp/benchmark `
  --attn-implementation sdpa `
  --packing on `
  --max-length 1024 `
  --num-train-epochs 1 `
  --task-upsample 1.5_state_tracking=4 `
  --task-upsample 1.9_fen_assembly=4 `
  --task-upsample 1.10_fen_row_application=4 `
  --skip-trainer-eval `
  --trainer-save-steps 1000 `
  --skip-eval `
  --no-acpl `
  --wandb-project chess-sft `
  --wandb-group phase-a-t12-v25000-20260701-decomp-pack1024 `
  --run-name phase-a-t12-v25000-20260701-decomp-pack1024-train
```

Expected: training writes `/home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-20260701-decomp-pack1024/phase_a/READY` and exports `phase_a/best`.

- [ ] **Step 2: Run vLLM eval on the second GPU**

```powershell
$env:WANDB_GIT_COMMIT = (git rev-parse HEAD).Trim()
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:VLLM_USE_V2_MODEL_RUNNER = '0'
$env:VLLM_WORKER_MULTIPROC_METHOD = 'spawn'

.\sft\training\run-wsl.ps1 `
  -NoSync `
  -CudaDeviceId 0 `
  -VenvPath /home/amazi/code/chess_sft_sdpo/.venv-vllm `
  -WslDataRoot /home/amazi/chess_sft_data/phase-a-t12-v25000-20260701-decomp `
  -WslCheckpointRoot /home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-20260701-decomp-pack1024 `
  -- chess-llm-evaluate `
  --model /home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-20260701-decomp-pack1024/phase_a/best `
  --benchmark-dir /home/amazi/chess_sft_data/phase-a-t12-v25000-20260701-decomp/benchmark `
  --output /home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-20260701-decomp-pack1024/phase_a/eval_predictions.vllm.jsonl `
  --phase a `
  --soft-gate `
  --max-examples-per-split 500 `
  --batch-size 32 `
  --max-new-tokens 192 `
  --attn-implementation auto `
  --inference-backend vllm `
  --vllm-gpu-memory-utilization 0.70 `
  --vllm-max-model-len 4096 `
  --no-acpl `
  --wandb-project chess-sft `
  --wandb-run-name phase-a-t12-v25000-20260701-decomp-pack1024-vllm-eval `
  --wandb-group phase-a-t12-v25000-20260701-decomp-pack1024
```

Expected: eval exits `0`, logs to W&B, and writes predictions/results next to the checkpoint.

- [ ] **Step 3: Record rehearsal results**

Collect the exact commit and result files:

```powershell
git rev-parse HEAD
.\sft\training\run-wsl.ps1 -NoSync -- bash -lc "ls -lh /home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-20260701-decomp-pack1024/phase_a && find /home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-20260701-decomp-pack1024/phase_a -maxdepth 1 -type f -name '*result*' -o -name '*analysis*'"
```

Then add a dated entry to `docs/experiments/experiment_log.md` containing these exact fields with measured values from W&B or the eval result JSON: git commit, data root, checkpoint root, W&B train run, W&B eval run, train runtime, input tokens/sec, total train tokens, perception overall, rules overall, state tracking, FEN assembly, legal move generation, piece legal filter, legal moves by piece, and the decision on whether to continue to full Phase A. Do not commit the log until each field has a concrete value.

### Task 5: Generate And Validate Full Phase A Data

**Files:**
- Create external artifact: `/home/amazi/chess_sft_data/phase-a-t12-full-20260701`
- Update after run: `docs/experiments/experiment_log.md`

- [ ] **Step 1: Generate full default tiers 1-2**

```powershell
.\sft\training\run-wsl.ps1 `
  -WslRepoPath /home/amazi/code/chess_sft_sdpo `
  -VenvPath /home/amazi/code/chess_sft_sdpo/.venv `
  -WslDataRoot /home/amazi/chess_sft_data/phase-a-t12-full-20260701 `
  -- chess-llm-make-data `
  --tier 1 2 `
  --source-readiness-report /home/amazi/chess_sft_data/phase-a-t12-full-20260701/readiness.json
```

Expected: 28 tier task files are generated using default volumes and the command exits `0`.

- [ ] **Step 2: Validate full data**

```powershell
.\sft\training\run-wsl.ps1 `
  -NoSync `
  -WslDataRoot /home/amazi/chess_sft_data/phase-a-t12-full-20260701 `
  -- chess-llm-make-data `
  --validate-only `
  --tier 1 2
```

Expected: validation exits `0` with no completeness or contamination errors.

- [ ] **Step 3: Dry-run the full training config**

```powershell
.\sft\training\run-wsl.ps1 `
  -NoSync `
  -WslDataRoot /home/amazi/chess_sft_data/phase-a-t12-full-20260701 `
  -WslCheckpointRoot /home/amazi/chess_sft_checkpoints/phase-a-t12-full-20260701-pack1024 `
  -- chess-llm-train `
  --phase a `
  --data-root /home/amazi/chess_sft_data/phase-a-t12-full-20260701/output `
  --output-root /home/amazi/chess_sft_checkpoints/phase-a-t12-full-20260701-pack1024 `
  --benchmark-dir /home/amazi/chess_sft_data/phase-a-t12-full-20260701/benchmark `
  --attn-implementation sdpa `
  --packing on `
  --max-length 1024 `
  --num-train-epochs 1 `
  --task-upsample 1.5_state_tracking=4 `
  --task-upsample 1.9_fen_assembly=4 `
  --task-upsample 1.10_fen_row_application=4 `
  --skip-trainer-eval `
  --trainer-save-steps 1000 `
  --skip-eval `
  --no-acpl `
  --wandb-project chess-sft `
  --wandb-group phase-a-t12-full-20260701-pack1024 `
  --run-name phase-a-t12-full-20260701-pack1024-dry-run `
  --dry-run
```

Expected: dry-run exits `0` and prints the true full-run optimizer step estimate.

### Task 6: Launch Full Phase A And Sidecar Eval

**Files:**
- Create external artifact: `/home/amazi/chess_sft_checkpoints/phase-a-t12-full-20260701-pack1024`
- Update after run: `docs/experiments/experiment_log.md`

- [ ] **Step 1: Launch full one-pass Phase A training**

```powershell
$env:WANDB_GIT_COMMIT = (git rev-parse HEAD).Trim()

.\sft\training\run-wsl.ps1 `
  -WslRepoPath /home/amazi/code/chess_sft_sdpo `
  -VenvPath /home/amazi/code/chess_sft_sdpo/.venv `
  -WslDataRoot /home/amazi/chess_sft_data/phase-a-t12-full-20260701 `
  -WslCheckpointRoot /home/amazi/chess_sft_checkpoints/phase-a-t12-full-20260701-pack1024 `
  -WslHfCache /home/amazi/.cache/huggingface `
  -WslWandbDir /home/amazi/chess_sft_wandb `
  -- chess-llm-train `
  --phase a `
  --data-root /home/amazi/chess_sft_data/phase-a-t12-full-20260701/output `
  --output-root /home/amazi/chess_sft_checkpoints/phase-a-t12-full-20260701-pack1024 `
  --benchmark-dir /home/amazi/chess_sft_data/phase-a-t12-full-20260701/benchmark `
  --attn-implementation sdpa `
  --packing on `
  --max-length 1024 `
  --num-train-epochs 1 `
  --task-upsample 1.5_state_tracking=4 `
  --task-upsample 1.9_fen_assembly=4 `
  --task-upsample 1.10_fen_row_application=4 `
  --skip-trainer-eval `
  --trainer-save-steps 1000 `
  --skip-eval `
  --no-acpl `
  --wandb-project chess-sft `
  --wandb-group phase-a-t12-full-20260701-pack1024 `
  --run-name phase-a-t12-full-20260701-pack1024-train
```

Expected: training writes `/home/amazi/chess_sft_checkpoints/phase-a-t12-full-20260701-pack1024/phase_a/READY`, exports `phase_a/best`, and logs online to W&B.

- [ ] **Step 2: Resume if interrupted**

Run the same command as Step 1 with this extra argument appended:

```text
--resume-from-checkpoint auto
```

Expected: the trainer resumes from the latest `checkpoint-N` under `/home/amazi/chess_sft_checkpoints/phase-a-t12-full-20260701-pack1024/phase_a`.

- [ ] **Step 3: Launch final vLLM sidecar eval**

```powershell
$env:WANDB_GIT_COMMIT = (git rev-parse HEAD).Trim()
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:VLLM_USE_V2_MODEL_RUNNER = '0'
$env:VLLM_WORKER_MULTIPROC_METHOD = 'spawn'

.\sft\training\run-wsl.ps1 `
  -NoSync `
  -CudaDeviceId 0 `
  -VenvPath /home/amazi/code/chess_sft_sdpo/.venv-vllm `
  -WslDataRoot /home/amazi/chess_sft_data/phase-a-t12-full-20260701 `
  -WslCheckpointRoot /home/amazi/chess_sft_checkpoints/phase-a-t12-full-20260701-pack1024 `
  -- chess-llm-evaluate `
  --model /home/amazi/chess_sft_checkpoints/phase-a-t12-full-20260701-pack1024/phase_a/best `
  --benchmark-dir /home/amazi/chess_sft_data/phase-a-t12-full-20260701/benchmark `
  --output /home/amazi/chess_sft_checkpoints/phase-a-t12-full-20260701-pack1024/phase_a/eval_predictions.vllm.jsonl `
  --phase a `
  --soft-gate `
  --batch-size 32 `
  --max-new-tokens 192 `
  --attn-implementation auto `
  --inference-backend vllm `
  --vllm-gpu-memory-utilization 0.70 `
  --vllm-max-model-len 4096 `
  --no-acpl `
  --wandb-project chess-sft `
  --wandb-run-name phase-a-t12-full-20260701-pack1024-vllm-eval `
  --wandb-group phase-a-t12-full-20260701-pack1024
```

Expected: eval exits `0`, logs to W&B, and writes prediction/result artifacts in `phase_a`.

### Task 7: Analyze And Decide The Next Curriculum Move

**Files:**
- Modify: `docs/experiments/experiment_log.md`
- Read: `/home/amazi/chess_sft_checkpoints/phase-a-t12-full-20260701-pack1024/phase_a/eval_predictions.vllm.jsonl`
- Read generated result/analysis JSON files next to the eval output

- [ ] **Step 1: Add full run entry to the experiment log**

Collect the exact commit and result files:

```powershell
git rev-parse HEAD
.\sft\training\run-wsl.ps1 -NoSync -- bash -lc "ls -lh /home/amazi/chess_sft_checkpoints/phase-a-t12-full-20260701-pack1024/phase_a && find /home/amazi/chess_sft_checkpoints/phase-a-t12-full-20260701-pack1024/phase_a -maxdepth 1 -type f -name '*result*' -o -name '*analysis*'"
```

Then append a dated entry to `docs/experiments/experiment_log.md` containing these exact fields with measured values from W&B, the training dry-run output, eval result JSON, and prediction analysis: git commit, data root, checkpoint root, W&B train run, W&B eval run, raw generated rows, effective rows after task upsampling, optimizer steps, train runtime, input tokens/sec, total train tokens, perception overall, rules overall, board to FEN, state tracking, FEN assembly, FEN row application, legal move generation, move legality, piece legal filter, legal moves by piece, main failure modes, and decision for next run. Do not commit the log until each field has a concrete value.

- [ ] **Step 2: Commit the experiment log**

```powershell
git add docs/experiments/experiment_log.md
git commit -m "docs: record full phase a sft run"
git push origin HEAD
```

Expected: experiment log is tied to the same repository history as the training run.

- [ ] **Step 3: State the next action**

Choose exactly one next action based on the eval:

```text
1. Continue to Phase B if FEN/state tracking improved and legal-move diagnostics are usable.
2. Add more legal-decomposition examples if legal move generation remains flat.
3. Add explicit reasoning-token SFT examples if format and mechanics are good but multi-ply reasoning is weak.
4. Reduce or rebalance upsampling if train throughput or output length is the bottleneck.
```

Record the chosen action in `docs/experiments/experiment_log.md` under `Decision for next run`.

## Self-Review

- Spec coverage: The plan covers code freeze, WSL preflight, W&B, data generation, validation, dry-run sizing, 25k rehearsal, full default Phase A, vLLM sidecar eval, resume behavior, and post-run analysis.
- Placeholder scan: Future experiment-log fields are described as measured values that must be collected before commit; the plan has no empty future-value holes.
- Type and command consistency: Commands use current entrypoints and current Phase A options: `--num-train-epochs 1`, `--skip-trainer-eval`, `--skip-eval`, task upsampling for `1.5`, `1.9`, and `1.10`, vLLM with `VLLM_USE_V2_MODEL_RUNNER=0`, and W&B project `chess-sft`.
