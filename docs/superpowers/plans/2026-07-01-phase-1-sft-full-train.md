# Phase 1 SFT Full Train Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the first full Phase 1 / Phase A SFT run to a launchable state, then train one pass over freshly generated tiers 1-2 data with W&B logging and vLLM sidecar eval.

**Architecture:** Keep data generation, training, and eval as separate artifacts with explicit Linux filesystem roots. The July 2 packed rehearsal proved that the mechanics work, but it also showed that the unchanged curriculum is inefficient on legal move filtering and material counting. Patch those targeted curriculum areas first, run one focused rehearsal to verify the signal, then generate the full tiers 1-2 data root, train one epoch with trainer eval disabled, and evaluate checkpoints with vLLM on the second GPU.

**Tech Stack:** Python package `chess_llm`, `chess-llm-make-data`, `chess-llm-train`, `chess-llm-evaluate`, Ubuntu-24.04-CUDA WSL, PyTorch, TRL/SFTTrainer, W&B, vLLM, PowerShell launch wrappers.

---

## Current Phase A Target

Current default volumes in `src/chess_llm/sft/settings.py`:

- Tier 1 rows: 1,520,000
- Tier 2 rows: 650,000
- Phase A raw rows: 2,170,000
- One-pass effective rows with `1.5`, `1.9`, and `1.10` upsampled to factor 4: about 3,070,000 rows before train/eval split

Use the trainer dry-run output as the source of truth for row counts, and use capped smokes on the exact data root as the source of truth for optimizer step time and wall-clock estimates. The current 2026-07-02 prelaunch path is `--attn-implementation sdpa --packing off --max-length 1024`; a controlled 100-step comparison on the refreshed legal/material data measured about 5.1k train tokens/sec unpacked versus about 1.2k train tokens/sec packed.

## Active Goal State - 2026-07-02

Codex goal:

```text
Get Phase 1 SFT ready for a full training run, execute the pre-run validation/rehearsal workflow, launch the one-pass Phase 1 training with W&B and sidecar eval, and analyze the run outputs to decide the next curriculum/training iteration.
```

Completed packed rehearsal:

- Commit: `8ddf3db415c9ab76ac894ba13a500ab5c5b28a74`
- Data root: `/home/amazi/chess_sft_data/phase-a-t12-v25000-20260701-decomp`
- Checkpoint root: `/home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-20260701-decomp-pack1024`
- Train run: `phase-a-t12-v25000-20260701-decomp-pack1024-train`
- Eval run: `phase-a-t12-v25000-20260701-decomp-pack1024-vllm-sidecar-eval`
- Train shape: one epoch, packed, max length 1024, 7,049 optimizer steps, 224,827,352 input tokens
- Train runtime: 10:35:30.91
- Train throughput: about 5,896 input tokens/sec
- Eval shape: vLLM sidecar, 500 perception examples, 500 rules examples, `--vllm-max-model-len 4096`

Rehearsal metrics:

| Split | Metric | Score |
| --- | --- | ---: |
| Perception | overall | 85.2% |
| Perception | board_to_fen | 96.4% |
| Perception | fen_assembly | 85.7% |
| Perception | fen_row_application | 75.0% |
| Perception | state_tracking | 82.1% |
| Perception | material_count | 25.0% |
| Rules | overall | 56.8% |
| Rules | legal_moves | 36.1% |
| Rules | piece_legal_moves | 51.2% |
| Rules | piece_legal_filter | 26.0% |
| Rules | legal_moves_by_piece | 0.0% |

Decision: do not launch the full Phase A pass on the unchanged recipe. The full-train goal stays active, but the next implementation step is targeted material-count and legal-move decomposition, then a smaller rehearsal to confirm that the model learns filtering and exact counting before the full run.

## Files And Artifacts

- Read: `docs/runbooks/phase_a_real_run.md`
- Read/update after runs: `docs/experiments/experiment_log.md`
- Read: `src/chess_llm/sft/settings.py`
- Read: `src/chess_llm/training/phases.py`
- Modify for the next curriculum patch: `src/chess_llm/sft/generators/tier1_perception.py`
- Modify for the next curriculum patch: `src/chess_llm/sft/generators/tier2_rules.py`
- Modify for prompt/format cleanup if needed: `src/chess_llm/sft/templates.py`
- Modify for volume/rebalance if needed: `src/chess_llm/sft/settings.py`
- Modify validation when answer formats change: `src/chess_llm/sft/validation.py`
- Test targeted curriculum changes: `tests/test_sft_generators.py`
- Test validation/scoring changes: `tests/test_sft_validation.py`, `tests/test_evals_benchmark.py`
- Use: `sft/training/run-wsl.ps1`
- Optional helper references: `.tmp/run_phase_a_t12_generate.ps1`, `.tmp/run_phase_a_t12_train.ps1`, `.tmp/run_phase_a_t12_vllm_sidecar_eval.ps1`
- Rehearsal data root: `/home/amazi/chess_sft_data/phase-a-t12-v25000-20260701-decomp`
- Rehearsal checkpoint root: `/home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-20260701-decomp-pack1024`
- Next targeted rehearsal data root: `/home/amazi/chess_sft_data/phase-a-t12-v25000-legal-material-20260702`
- Next targeted rehearsal checkpoint root: `/home/amazi/chess_sft_checkpoints/phase-a-t12-v25000-legal-material-20260702-sdpa-unpacked`
- Full data root: `/home/amazi/chess_sft_data/phase-a-t12-full-legal-material-20260702`
- Full checkpoint root: `/home/amazi/chess_sft_checkpoints/phase-a-t12-full-legal-material-20260702-sdpa-unpacked`

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
  -- chess-llm-train --phase a --attn-implementation sdpa --packing off --max-length 1024 --num-train-epochs 1 --skip-trainer-eval --skip-eval --dry-run
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
  --packing off `
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
  --packing off `
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

`chess-llm-evaluate` auto-exports local Qwen3.5 text-only SFT checkpoints to
`best/vllm_qwen35_wrapper` before constructing vLLM, so keep `--model` pointed
at the normal `phase_a/best` directory.

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
  --max-new-tokens 384 `
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

Then add a dated entry to `docs/experiments/experiment_log.md` containing these exact fields with measured values from W&B or the eval result JSON: git commit, data root, checkpoint root, W&B train run, W&B eval run, train runtime, input tokens/sec, total train tokens, perception overall, rules overall, state tracking, FEN assembly, legal move generation, piece legal filter, legal moves by piece exact score, legal moves by piece partial diagnostics (`all_moves_jaccard`, `all_moves_precision`, `all_moves_recall`, `per_piece_group_jaccard`, `section_completeness`, `illegal_extra_count`, `missing_move_count`), and the decision on whether to continue to full Phase A. Do not commit the log until each field has a concrete value.

### Task 5: Patch Targeted Material And Legal Curriculum

**Files:**
- Modify: `src/chess_llm/sft/generators/tier1_perception.py`
- Modify: `src/chess_llm/sft/generators/tier2_rules.py`
- Modify if answer formats change: `src/chess_llm/sft/templates.py`
- Modify if validation expectations change: `src/chess_llm/sft/validation.py`
- Modify if volume mix changes: `src/chess_llm/sft/settings.py`
- Test: `tests/test_sft_generators.py`
- Test: `tests/test_sft_validation.py`
- Test: `tests/test_evals_benchmark.py`

- [ ] **Step 1: Add material-count decomposition tests**

Add generator tests that assert material examples force the intermediate state the model is currently skipping:

```python
def test_package_material_balance_trace_uses_balance_label():
    from random import Random

    from chess_llm.sft.generators.tier1_perception import MaterialBalanceTrace

    row = next(
        MaterialBalanceTrace(
            config={
                "fen_pool": [{"fen": "8/8/8/8/8/8/6p1/4K2k w - - 0 1"}],
                "volume_override": 1,
            },
            rng=Random(0),
        ).generate()
    )
    answer = row["messages"][2]["content"]

    assert answer.startswith("Inventory: ")
    assert "\nCounts: " in answer
    assert "\nValues: " in answer
    assert "\nBalance: " in answer
    assert "\nFinal: " not in answer
```

Run:

```powershell
python -m pytest tests/test_sft_generators.py::test_package_material_balance_trace_uses_balance_label -q
```

Expected: fail until the answer format has the explicit staged fields.

- [ ] **Step 2: Add legal-filter decomposition tests**

Add generator tests that assert the model sees rejected pseudo-legal moves with explicit reasons instead of only final legal move sets:

```python
def test_package_piece_legal_filter_includes_rejected_reason_labels():
    from random import Random

    from chess_llm.sft.generators.tier2_rules import PieceLegalFilter

    row = next(
        PieceLegalFilter(
            config={
                "fen_pool": [{"fen": "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1"}],
                "volume_override": 1,
            },
            rng=Random(0),
        ).generate()
    )
    answer = row["messages"][2]["content"]

    assert "Pseudo-legal:" in answer
    assert "Legal:" in answer
    assert "Rejected:" in answer
    assert "pinned_piece_exposes_king" in answer
```

Run:

```powershell
python -m pytest tests/test_sft_generators.py::test_package_piece_legal_filter_includes_rejected_reason_labels -q
```

Expected: fail while rejected moves are still reasonless.

- [ ] **Step 3: Implement the curriculum patch**

Keep the answer formats short and mechanical:

```text
Inventory: white K=1 Q=0 R=2 B=0 N=1 P=5 | black K=1 Q=1 R=0 B=1 N=0 P=6
Counts: white pieces=9 value=18 | black pieces=9 value=18
Balance: equal material
```

```text
Pseudo-legal: h1h2, h1h3, h1e1.
Legal: h1h2, h1h3.
Rejected: h1e1 pinned_piece_exposes_king.
```

Do not add long prose reasoning. The target is reliable lookup, filtering, counting, and exact compact output.

- [ ] **Step 4: Run focused tests**

```powershell
python -m pytest tests/test_sft_generators.py tests/test_sft_validation.py tests/test_evals_benchmark.py -q
```

Expected: all targeted generator, validation, and eval tests pass.

- [ ] **Step 5: Generate a targeted 25k rehearsal**

```powershell
.\sft\training\run-wsl.ps1 `
  -WslRepoPath /home/amazi/code/chess_sft_sdpo `
  -VenvPath /home/amazi/code/chess_sft_sdpo/.venv `
  -WslDataRoot /home/amazi/chess_sft_data/phase-a-t12-v25000-legal-material-20260702 `
  -- chess-llm-make-data `
  --tier 1 2 `
  --volume 25000 `
  --eval-split-volume 25000 `
  --source-readiness-report /home/amazi/chess_sft_data/phase-a-t12-v25000-legal-material-20260702/readiness.json
```

Expected: generation exits `0` and validation can be run on the new root.

- [ ] **Step 6: Run an unpacked SDPA rehearsal before full launch**

Use the current unpacked SDPA recipe, but point it at `/home/amazi/chess_sft_data/phase-a-t12-v25000-legal-material-20260702` and a fresh checkpoint root for the legal/material rehearsal.

Expected: W&B logs train throughput and vLLM sidecar eval writes result JSON. The key question is whether `material_count`, `piece_legal_filter`, `legal_moves`, and `legal_moves_by_piece` improve relative to the July 2 rehearsal without regressing FEN/state mechanics.

### Task 6: Generate And Validate Full Phase A Data

**Files:**
- Create external artifact: `/home/amazi/chess_sft_data/phase-a-t12-full-legal-material-20260702`
- Update after run: `docs/experiments/experiment_log.md`

- [ ] **Step 1: Generate full default tiers 1-2**

```powershell
.\sft\training\run-wsl.ps1 `
  -WslRepoPath /home/amazi/code/chess_sft_sdpo `
  -VenvPath /home/amazi/code/chess_sft_sdpo/.venv `
  -WslDataRoot /home/amazi/chess_sft_data/phase-a-t12-full-legal-material-20260702 `
  -- chess-llm-make-data `
  --tier 1 2 `
  --source-readiness-report /home/amazi/chess_sft_data/phase-a-t12-full-legal-material-20260702/readiness.json
```

Expected: 28 tier task files are generated using default volumes and the command exits `0`.

- [ ] **Step 2: Validate full data**

```powershell
.\sft\training\run-wsl.ps1 `
  -NoSync `
  -WslDataRoot /home/amazi/chess_sft_data/phase-a-t12-full-legal-material-20260702 `
  -- chess-llm-make-data `
  --validate-only `
  --tier 1 2
```

Expected: validation exits `0` with no completeness or contamination errors.

- [ ] **Step 3: Dry-run the full training config**

```powershell
.\sft\training\run-wsl.ps1 `
  -NoSync `
  -WslDataRoot /home/amazi/chess_sft_data/phase-a-t12-full-legal-material-20260702 `
  -WslCheckpointRoot /home/amazi/chess_sft_checkpoints/phase-a-t12-full-legal-material-20260702-sdpa-unpacked `
  -- chess-llm-train `
  --phase a `
  --data-root /home/amazi/chess_sft_data/phase-a-t12-full-legal-material-20260702/output `
  --output-root /home/amazi/chess_sft_checkpoints/phase-a-t12-full-legal-material-20260702-sdpa-unpacked `
  --benchmark-dir /home/amazi/chess_sft_data/phase-a-t12-full-legal-material-20260702/benchmark `
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
  --no-acpl `
  --wandb-project chess-sft `
  --wandb-group phase-a-t12-full-legal-material-20260702-sdpa-unpacked `
  --run-name phase-a-t12-full-legal-material-20260702-sdpa-unpacked-dry-run `
  --dry-run
```

Expected: dry-run exits `0` and prints the true full-run optimizer step estimate.

### Task 7: Launch Full Phase A And Sidecar Eval

**Files:**
- Create external artifact: `/home/amazi/chess_sft_checkpoints/phase-a-t12-full-legal-material-20260702-sdpa-unpacked`
- Update after run: `docs/experiments/experiment_log.md`

- [ ] **Step 1: Launch full one-pass Phase A training**

```powershell
$env:WANDB_GIT_COMMIT = (git rev-parse HEAD).Trim()

.\sft\training\run-wsl.ps1 `
  -WslRepoPath /home/amazi/code/chess_sft_sdpo `
  -VenvPath /home/amazi/code/chess_sft_sdpo/.venv `
  -WslDataRoot /home/amazi/chess_sft_data/phase-a-t12-full-legal-material-20260702 `
  -WslCheckpointRoot /home/amazi/chess_sft_checkpoints/phase-a-t12-full-legal-material-20260702-sdpa-unpacked `
  -WslHfCache /home/amazi/.cache/huggingface `
  -WslWandbDir /home/amazi/chess_sft_wandb `
  -- chess-llm-train `
  --phase a `
  --data-root /home/amazi/chess_sft_data/phase-a-t12-full-legal-material-20260702/output `
  --output-root /home/amazi/chess_sft_checkpoints/phase-a-t12-full-legal-material-20260702-sdpa-unpacked `
  --benchmark-dir /home/amazi/chess_sft_data/phase-a-t12-full-legal-material-20260702/benchmark `
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
  --no-acpl `
  --wandb-project chess-sft `
  --wandb-group phase-a-t12-full-legal-material-20260702-sdpa-unpacked `
  --run-name phase-a-t12-full-legal-material-20260702-sdpa-unpacked-train
```

Expected: training writes `/home/amazi/chess_sft_checkpoints/phase-a-t12-full-legal-material-20260702-sdpa-unpacked/phase_a/READY`, exports `phase_a/best`, and logs online to W&B.

- [ ] **Step 2: Resume if interrupted**

Run the same command as Step 1 with this extra argument appended:

```text
--resume-from-checkpoint auto
```

Expected: the trainer resumes from the latest `checkpoint-N` under `/home/amazi/chess_sft_checkpoints/phase-a-t12-full-legal-material-20260702-sdpa-unpacked/phase_a`.

- [ ] **Step 3: Launch final vLLM sidecar eval**

`chess-llm-evaluate` auto-exports local Qwen3.5 text-only SFT checkpoints to
`best/vllm_qwen35_wrapper` before constructing vLLM, so keep `--model` pointed
at the normal `phase_a/best` directory.

```powershell
$env:WANDB_GIT_COMMIT = (git rev-parse HEAD).Trim()
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:VLLM_USE_V2_MODEL_RUNNER = '0'
$env:VLLM_WORKER_MULTIPROC_METHOD = 'spawn'

.\sft\training\run-wsl.ps1 `
  -NoSync `
  -CudaDeviceId 0 `
  -VenvPath /home/amazi/code/chess_sft_sdpo/.venv-vllm `
  -WslDataRoot /home/amazi/chess_sft_data/phase-a-t12-full-legal-material-20260702 `
  -WslCheckpointRoot /home/amazi/chess_sft_checkpoints/phase-a-t12-full-legal-material-20260702-sdpa-unpacked `
  -- chess-llm-evaluate `
  --model /home/amazi/chess_sft_checkpoints/phase-a-t12-full-legal-material-20260702-sdpa-unpacked/phase_a/best `
  --benchmark-dir /home/amazi/chess_sft_data/phase-a-t12-full-legal-material-20260702/benchmark `
  --output /home/amazi/chess_sft_checkpoints/phase-a-t12-full-legal-material-20260702-sdpa-unpacked/phase_a/eval_predictions.vllm.jsonl `
  --phase a `
  --soft-gate `
  --batch-size 32 `
  --max-new-tokens 384 `
  --attn-implementation auto `
  --inference-backend vllm `
  --vllm-gpu-memory-utilization 0.70 `
  --vllm-max-model-len 4096 `
  --no-acpl `
  --wandb-project chess-sft `
  --wandb-run-name phase-a-t12-full-legal-material-20260702-sdpa-unpacked-vllm-eval `
  --wandb-group phase-a-t12-full-legal-material-20260702-sdpa-unpacked
```

Expected: eval exits `0`, logs to W&B, and writes prediction/result artifacts in `phase_a`.

### Task 8: Analyze And Decide The Next Curriculum Move

**Files:**
- Modify: `docs/experiments/experiment_log.md`
- Read: `/home/amazi/chess_sft_checkpoints/phase-a-t12-full-legal-material-20260702-sdpa-unpacked/phase_a/eval_predictions.vllm.jsonl`
- Read generated result/analysis JSON files next to the eval output

- [ ] **Step 1: Add full run entry to the experiment log**

Collect the exact commit and result files:

```powershell
git rev-parse HEAD
.\sft\training\run-wsl.ps1 -NoSync -- bash -lc "ls -lh /home/amazi/chess_sft_checkpoints/phase-a-t12-full-legal-material-20260702-sdpa-unpacked/phase_a && find /home/amazi/chess_sft_checkpoints/phase-a-t12-full-legal-material-20260702-sdpa-unpacked/phase_a -maxdepth 1 -type f -name '*result*' -o -name '*analysis*'"
```

Then append a dated entry to `docs/experiments/experiment_log.md` containing these exact fields with measured values from W&B, the training dry-run output, eval result JSON, and prediction analysis: git commit, data root, checkpoint root, W&B train run, W&B eval run, raw generated rows, effective rows after task upsampling, optimizer steps, train runtime, input tokens/sec, total train tokens, perception overall, rules overall, board to FEN, state tracking, FEN assembly, FEN row application, legal move generation, move legality, piece legal filter, legal moves by piece exact score, legal moves by piece partial diagnostics (`all_moves_jaccard`, `all_moves_precision`, `all_moves_recall`, `per_piece_group_jaccard`, `section_completeness`, `illegal_extra_count`, `missing_move_count`), main failure modes, and decision for next run. Do not commit the log until each field has a concrete value.

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

- Spec coverage: The plan covers code freeze, WSL preflight, W&B, the completed July 2 packed rehearsal, targeted material/legal curriculum cleanup, refreshed 25k rehearsal, full Phase A data generation, vLLM sidecar eval, resume behavior, and post-run analysis.
- Placeholder scan: Future experiment-log fields are described as measured values that must be collected before commit; the plan has no empty future-value holes.
- Type and command consistency: Commands use current entrypoints and current Phase A options: `--num-train-epochs 1`, `--skip-trainer-eval`, `--skip-eval`, task upsampling for `1.5`, `1.9`, and `1.10`, vLLM with `VLLM_USE_V2_MODEL_RUNNER=0`, and W&B project `chess-sft`.
