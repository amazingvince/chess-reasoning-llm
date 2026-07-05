# Phase A Legal-Focused Real Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:verification-before-completion` before claiming any stage complete. Use `superpowers:systematic-debugging` if generation, validation, training, or eval fails unexpectedly.

**Goal:** Run the next serious Phase A training experiment on fixed 31-task data with the curriculum mixer enabled, focused on closing the legal-move and legality gaps before spending the full 2.3M-row default budget.

**Architecture:** Generate a 25k-per-task Phase A dataset in the CUDA WSL training checkout, validate decontamination and completeness, dry-run the mixed curriculum, train one epoch with SDPA and no trainer eval, then run capped prediction eval with a larger generation budget. Use the result to decide whether to scale to the full default Phase A run or adjust generators/mix again.

**Tech Stack:** Python package `chess_llm`, WSL distro `Ubuntu-24.04-CUDA`, WSL checkout `/home/amazi/code/chess_sft_sdpo`, local `.venv` for training/eval, optional `.venv-vllm` for faster eval, W&B project `chess-sft`, Windows git checkout commit `40ea18eebe12dcb77882b3df11902109786ba186`.

---

## Current Evidence

The fixed generator commit produced a clean 62,000-row rehearsal dataset at `/home/amazi/chess_sft_data/phase-a-r10-v2000b`.

One-epoch Phase A rehearsal from checkpoint 500 finished successfully:

- Train W&B run: `https://wandb.ai/amazingvince/chess-sft/runs/z3cq3k9v`
- Eval W&B run: `https://wandb.ai/amazingvince/chess-sft/runs/q71gzkle`
- Results: `/home/amazi/chess_sft_checkpoints/phase-a-r10-v2000b/phase_a/eval_1epoch_500_predictions.results.json`
- Perception overall: `77.6`
- Rules overall: `56.5`
- `1.5_state_tracking`: `88.5`, now close enough that old 4x state emphasis should be reduced
- `1.9_fen_assembly`: `96.2`
- `1.10_fen_row_application`: `84.6`, first4 `96.2`
- `2.1_legal_move_gen`: `51.3`
- `2.3_move_legality_check`: `84.2`
- `2.11_legal_filter_trace`: exact `0.0`, final Jaccard `40.1`
- `2.9_legal_moves_by_piece`: exact `0.0`, set F1 `73.3`, group Jaccard `70.7`
- Weakest task: `rules/captures` at `17.9`

Conclusion: the next run should spend data and upsampling on legal generation, legal filtering, king safety, ray walk, and multi-move state. Do not launch the full default 2.3M-row run until the 25k run shows rules metrics are moving.

---

## Run Identity

- Dataset root: `/home/amazi/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703`
- Checkpoint root: `/home/amazi/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-20260703`
- W&B group: `phase-a-r10-v25000-legalfocus`
- Train run name: `phase-a-r10-v25000-legalfocus-train`
- Eval run name, transformers fallback: `phase-a-r10-v25000-legalfocus-eval-transformers-768`
- Eval run name, vLLM preferred: `phase-a-r10-v25000-legalfocus-eval-vllm-768`
- Expected raw dataset size: `31 * 25000 = 775000` examples
- Training policy: one epoch only, skip trainer eval, skip built-in post-train eval
- Eval policy: capped 500 examples per split first, `--max-new-tokens 768`, soft-gate/report mode so known gate failures do not mask metric output

---

## Curriculum Mixer

Use this mix for the dry-run and training command:

```text
--task-upsample 1.19_multi_move_state_tracking=3
--task-upsample 1.5_state_tracking=2
--task-upsample 1.9_fen_assembly=2
--task-upsample 1.10_fen_row_application=2
--task-upsample 2.1_legal_move_gen=3
--task-upsample 2.3_move_legality_check=2
--task-upsample 2.7_piece_legal_filter=4
--task-upsample 2.8_king_safety_filter=2
--task-upsample 2.9_legal_moves_by_piece=2
--task-upsample 2.10_ray_walk=2
--task-upsample 2.11_legal_filter_trace=4
```

Rationale:

- Reduce old state/FEN 4x pressure because state tracking and FEN assembly are no longer the bottleneck.
- Keep `1.19` high because multi-move state is still weak and can poison legal reasoning.
- Put the largest emphasis on `2.7` and `2.11`, where the model needs to learn to filter pseudo-legal moves and produce faithful legal-filter traces.
- Include `2.9` despite exact-match brittleness because partial set metrics are promising and need longer eval output budgets.

---

## Acceptance Criteria

The 25k run is worth scaling to the full default Phase A run only if the capped eval shows:

- `1.5_state_tracking >= 88` with no major regression from the v2000 run
- `1.9_fen_assembly >= 92`
- `1.10_fen_row_application >= 85`
- `2.1_legal_move_gen >= 70`
- `2.3_move_legality_check >= 88`
- `2.11_legal_filter_trace` final Jaccard `>= 60`
- `2.9_legal_moves_by_piece` set F1 `>= 80`
- Rules overall `>= 65`

If `2.1_legal_move_gen < 70`, do not scale yet. Inspect legal-move output failures and adjust either task formatting, decoding budget, or the legal-task mix.

If `2.3_move_legality_check < 88`, do not scale yet. Add targeted data around pinned pieces, king exposure, captures, special rules, and occupied-square edge cases.

If perception tasks regress while rules improve, reduce legal upsampling by one tier and restore `1.10_fen_row_application=3`.

---

## Task 1: Confirm Clean Baseline

Run from Windows PowerShell:

```powershell
git -C C:\Users\amazi\code\chess_sft_sdpo status --short
git -C C:\Users\amazi\code\chess_sft_sdpo rev-parse HEAD
```

Expected:

- `git status --short` has no unexpected changes that would contaminate the run.
- `git rev-parse HEAD` prints `40ea18eebe12dcb77882b3df11902109786ba186`.

If the worktree is dirty, either commit the intended changes or record the exact diff before running.

---

## Task 2: Generate 25k-Per-Task Dataset

Run from Windows PowerShell:

```powershell
wsl -d Ubuntu-24.04-CUDA --cd /home/amazi/code/chess_sft_sdpo -- bash -lc 'set -euo pipefail
source .venv/bin/activate
export PYTHONHASHSEED=0
export CHESS_SFT_OUTPUT=/home/amazi/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703
mkdir -p "$CHESS_SFT_OUTPUT"
chess-llm-make-data \
  --tier 1 2 \
  --volume 25000 \
  --source-readiness-report /home/amazi/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/readiness.json
'
```

Expected:

- `31` output JSONL files under `/home/amazi/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/output`
- `25000` examples per task file
- `readiness.json` exists

---

## Task 3: Validate Dataset

Run:

```powershell
wsl -d Ubuntu-24.04-CUDA --cd /home/amazi/code/chess_sft_sdpo -- bash -lc 'set -euo pipefail
source .venv/bin/activate
chess-llm-validate-outputs \
  --output-dir /home/amazi/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/output \
  --blocklist-path /home/amazi/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/eval_splits/blocklist.txt \
  --expected-volume 25000 \
  --tier 1 2
'
```

Expected:

- Completeness passes for all 31 tasks.
- Decontamination passes.
- There are no formatter, parse, or legality validation errors.

If the blocklist path is produced by a split-building step in the local CLI instead of generation, run the repo's existing split command first and then rerun validation against the produced blocklist.

---

## Task 4: Dry-Run Curriculum Mixer

Run:

```powershell
wsl -d Ubuntu-24.04-CUDA --cd /home/amazi/code/chess_sft_sdpo -- bash -lc 'set -euo pipefail
source .venv/bin/activate
WANDB_MODE=disabled chess-llm-train \
  --phase a \
  --data-root /home/amazi/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/output \
  --output-root /home/amazi/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-20260703 \
  --base-model Qwen/Qwen3.5-0.8B \
  --attn-implementation sdpa \
  --max-length 1024 \
  --packing off \
  --num-train-epochs 1 \
  --skip-trainer-eval \
  --skip-eval \
  --dry-run \
  --task-upsample 1.19_multi_move_state_tracking=3 \
  --task-upsample 1.5_state_tracking=2 \
  --task-upsample 1.9_fen_assembly=2 \
  --task-upsample 1.10_fen_row_application=2 \
  --task-upsample 2.1_legal_move_gen=3 \
  --task-upsample 2.3_move_legality_check=2 \
  --task-upsample 2.7_piece_legal_filter=4 \
  --task-upsample 2.8_king_safety_filter=2 \
  --task-upsample 2.9_legal_moves_by_piece=2 \
  --task-upsample 2.10_ray_walk=2 \
  --task-upsample 2.11_legal_filter_trace=4
'
```

Record:

- Final train row count after split and upsampling
- Eval row count
- Per-task pre-upsample and post-upsample counts
- Whether any task is accidentally dominating the mix

Abort before full training if the mixer produces a row count that is too large for a one-epoch run on available wall time. In that case, reduce `2.11_legal_filter_trace` from `4` to `3` and rerun the dry-run.

---

## Task 5: Launch One-Epoch Training

Run:

```powershell
wsl -d Ubuntu-24.04-CUDA --cd /home/amazi/code/chess_sft_sdpo -- bash -lc 'set -euo pipefail
source .venv/bin/activate
export WANDB_PROJECT=chess-sft
export WANDB_GROUP=phase-a-r10-v25000-legalfocus
export WANDB_RUN_NAME=phase-a-r10-v25000-legalfocus-train
export WANDB_GIT_COMMIT=40ea18eebe12dcb77882b3df11902109786ba186
chess-llm-train \
  --phase a \
  --data-root /home/amazi/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/output \
  --output-root /home/amazi/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-20260703 \
  --base-model Qwen/Qwen3.5-0.8B \
  --attn-implementation sdpa \
  --max-length 1024 \
  --packing off \
  --num-train-epochs 1 \
  --skip-trainer-eval \
  --skip-eval \
  --trainer-save-steps 1000 \
  --task-upsample 1.19_multi_move_state_tracking=3 \
  --task-upsample 1.5_state_tracking=2 \
  --task-upsample 1.9_fen_assembly=2 \
  --task-upsample 1.10_fen_row_application=2 \
  --task-upsample 2.1_legal_move_gen=3 \
  --task-upsample 2.3_move_legality_check=2 \
  --task-upsample 2.7_piece_legal_filter=4 \
  --task-upsample 2.8_king_safety_filter=2 \
  --task-upsample 2.9_legal_moves_by_piece=2 \
  --task-upsample 2.10_ray_walk=2 \
  --task-upsample 2.11_legal_filter_trace=4
'
```

Expected:

- Checkpoints are written under `/home/amazi/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-20260703/phase_a`
- Final or best adapter is marked ready by the existing training flow
- W&B has one train run in group `phase-a-r10-v25000-legalfocus`

Resume command if interrupted:

```powershell
wsl -d Ubuntu-24.04-CUDA --cd /home/amazi/code/chess_sft_sdpo -- bash -lc 'set -euo pipefail
source .venv/bin/activate
export WANDB_PROJECT=chess-sft
export WANDB_GROUP=phase-a-r10-v25000-legalfocus
export WANDB_RUN_NAME=phase-a-r10-v25000-legalfocus-train-resume
export WANDB_GIT_COMMIT=40ea18eebe12dcb77882b3df11902109786ba186
chess-llm-train \
  --phase a \
  --data-root /home/amazi/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/output \
  --output-root /home/amazi/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-20260703 \
  --base-model Qwen/Qwen3.5-0.8B \
  --attn-implementation sdpa \
  --max-length 1024 \
  --packing off \
  --num-train-epochs 1 \
  --skip-trainer-eval \
  --skip-eval \
  --trainer-save-steps 1000 \
  --resume-from-checkpoint auto \
  --task-upsample 1.19_multi_move_state_tracking=3 \
  --task-upsample 1.5_state_tracking=2 \
  --task-upsample 1.9_fen_assembly=2 \
  --task-upsample 1.10_fen_row_application=2 \
  --task-upsample 2.1_legal_move_gen=3 \
  --task-upsample 2.3_move_legality_check=2 \
  --task-upsample 2.7_piece_legal_filter=4 \
  --task-upsample 2.8_king_safety_filter=2 \
  --task-upsample 2.9_legal_moves_by_piece=2 \
  --task-upsample 2.10_ray_walk=2 \
  --task-upsample 2.11_legal_filter_trace=4
'
```

---

## Task 6: Evaluate With Transformers Fallback

Use this if vLLM is not warmed or if the vLLM environment fails.

```powershell
wsl -d Ubuntu-24.04-CUDA --cd /home/amazi/code/chess_sft_sdpo -- bash -lc 'set -euo pipefail
source .venv/bin/activate
export WANDB_PROJECT=chess-sft
export WANDB_GROUP=phase-a-r10-v25000-legalfocus
export WANDB_RUN_NAME=phase-a-r10-v25000-legalfocus-eval-transformers-768
chess-llm-evaluate \
  --model /home/amazi/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-20260703/phase_a/best \
  --benchmark-dir /home/amazi/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/benchmark \
  --output /home/amazi/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-20260703/phase_a/eval_25k_768_predictions.jsonl \
  --phase a \
  --max-examples-per-split 500 \
  --max-new-tokens 768 \
  --batch-size 16 \
  --attn-implementation sdpa \
  --no-acpl \
  --soft-gate \
  --wandb-project chess-sft \
  --wandb-run-name phase-a-r10-v25000-legalfocus-eval-transformers-768
'
```

Expected:

- Eval artifacts are written under `/home/amazi/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-20260703/phase_a`
- The command exits successfully even if gates fail, because `--soft-gate` is set
- Metrics include exact and partial task metrics for `2.9` and `2.11`

---

## Task 7: Evaluate With vLLM Preferred Path

Use this after confirming `.venv-vllm` is healthy.

```powershell
wsl -d Ubuntu-24.04-CUDA --cd /home/amazi/code/chess_sft_sdpo -- bash -lc 'set -euo pipefail
source .venv-vllm/bin/activate
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_PROJECT=chess-sft
export WANDB_GROUP=phase-a-r10-v25000-legalfocus
export WANDB_RUN_NAME=phase-a-r10-v25000-legalfocus-eval-vllm-768
chess-llm-evaluate \
  --model /home/amazi/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-20260703/phase_a/best \
  --benchmark-dir /home/amazi/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/benchmark \
  --output /home/amazi/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-20260703/phase_a/eval_25k_768_predictions.vllm.jsonl \
  --phase a \
  --inference-backend vllm \
  --vllm-max-model-len 4096 \
  --max-examples-per-split 500 \
  --max-new-tokens 768 \
  --batch-size 32 \
  --no-acpl \
  --soft-gate \
  --wandb-project chess-sft \
  --wandb-run-name phase-a-r10-v25000-legalfocus-eval-vllm-768
'
```

If vLLM fails for environment reasons, do not debug vLLM during the training run unless transformers eval is too slow. Fall back to Task 6 and keep the experiment moving.

---

## Task 8: Update Experiment Log

Append a compact entry to `/mnt/c/Users/amazi/code/chess_sft_sdpo/docs/experiments/experiment_log.md` with:

- Git commit
- Dataset root
- Checkpoint root
- Training W&B URL
- Eval W&B URL
- Train row count after curriculum mixing
- Runtime
- Final train loss
- Tokens seen
- Phase A overall metrics
- Key task metrics listed in the acceptance criteria
- Decision: scale full default run, adjust mix, or fix generator/eval bottleneck

---

## Task 9: Decision Tree After Eval

Use the acceptance criteria to pick exactly one next action:

1. Scale to full default Phase A if rules and perception both improve enough.
2. Keep 25k data but run a second epoch only if train loss is still dropping and legal metrics are close to the thresholds.
3. Adjust the curriculum mix if legal tasks improve but perception regresses.
4. Fix generator or scorer behavior if `2.9`/`2.11` exact metrics remain zero while partial metrics are strong.
5. Add targeted legal examples if `2.1` or `2.3` remains the bottleneck.

Do not start Phase B, DPO, GRPO, or seeded reasoning generation from this checkpoint until Phase A legal metrics are stable enough to produce reliable traces.

---

## Verification Checklist

- [ ] `git rev-parse HEAD` equals `40ea18eebe12dcb77882b3df11902109786ba186`
- [ ] 31 generated task files exist
- [ ] each task has exactly 25,000 examples before splitting/mixing
- [ ] validation passes completeness
- [ ] validation passes decontamination
- [ ] curriculum dry-run logs train/eval row counts and task upsampling
- [ ] one-epoch training reaches final save
- [ ] eval runs with `--max-new-tokens 768`
- [ ] eval metrics include legal partial metrics
- [ ] experiment log is updated
- [ ] scale/no-scale decision is written down
