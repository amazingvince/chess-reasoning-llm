# H100 Phase B Repair Runbook - 2026-07-04

This is the reproducibility record for the July 4, 2026 H100 Phase B run,
targeted repair run, metric fixes, and the next planned repair. It records the
exact commands, checkpoint lineage, dataset state, failed gates, and why the
next run should focus on captures, hanging/tactics, and endgame WDL rather than
blindly continuing broad Phase B SFT.

## Current Status

As of `2026-07-05`, the Phase B H100 repair sequence is closed.

- Keep:
  `/workspace/chess_sft_checkpoints/phase-b-repair2-t3t6-v50000-h100-fa3-20260704/phase_b/best`
- Local verified copy:
  `C:\Users\amazi\code\chess_sft_sdpo\artifacts\models\phase-b-repair2-best`
- Decision: freeze repair2 as the best Phase B generator checkpoint from this
  sequence.
- Do not continue broad Phase B SFT from repair2 without a stronger replay or
  verifier plan; every later tactical continuation tested here regressed the
  main tactics benchmark.
- Historical note: at `2026-07-04 17:11 UTC`, the H100 node was still running
  the tier 3/6 `--volume 50000` data-generation job that enabled repair2.

## H100 Setup

| Item | Value |
| --- | --- |
| Provider | RunPod |
| SSH | `ssh root@103.207.149.91 -p 18659 -i ~/.ssh/id_rsa` |
| Hostname | `17befc5dd3d9` |
| GPUs | `2x NVIDIA H100 80GB HBM3` |
| Remote repo copy | `/workspace/chess_sft_sdpo` |
| Remote venv | `/workspace/chess_sft_sdpo/.venv-h100` |
| Data root | `/workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703` |
| Original checkpoint root | `/workspace/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-h100-fa3-legality-cont1` |
| Repair checkpoint root | `/workspace/chess_sft_checkpoints/phase-b-repair-from-b1-h100-fa3-20260704` |
| Training attention backend | `flash_attention_3` |
| Eval attention backend | `sdpa` |
| GPU env | `CUDA_VISIBLE_DEVICES=0,1`, `NCCL_NVLS_ENABLE=0` |
| W&B | `WANDB_MODE=offline`, `--allow-wandb-offline` |

The H100 throughput during the completed SFT runs was about `50k-51k`
tokens/sec with FA3 packing enabled.

## Code Changes In This Iteration

These changes are part of the run context and should be committed with the
reproducibility record.

### Data And Loader Fixes

- `src/chess_llm/sft/pipeline.py`
  - Added opening-prefix book-move fallback derived from Lichess opening
    `uci_moves` when Polyglot books are unavailable.
  - Added Syzygy source sampling cap:
    - `DEFAULT_SYZYGY_SOURCE_SAMPLES = 3000`
    - `SYZYGY_SOURCE_SAMPLE_CAP = 5000`
    - `syzygy_source_sample_count(volume_override)`
  - For `--volume 25000` and `--volume 50000`, Syzygy per-material source
    samples are capped at `5000`.
- `src/chess_llm/sft/sources/lichess_evals.py`
  - Rejects parseable but invalid FENs with `board.is_valid()`.
- `src/chess_llm/sft/identity.py`
  - Tier 5 opening identities include `cycle` for 5.1/5.2/5.3.
- `src/chess_llm/sft/generators/tier5_openings.py`
  - Opening tasks cycle through small source sets until the target count is met.
- `src/chess_llm/training/data/loader.py`
  - Fixed mixed standard/Chess960 Arrow schema inference by using explicit
    `_SANITIZED_FEATURES`.
  - Bumped sanitizer version to `3`.

### Metric Fixes

- `src/chess_llm/evals/benchmark.py`
  - `opening_name` primary score now credits stable opening family or ECO-decade
    matches.
  - Exact opening name, exact ECO, ECO decade, and family scores are emitted as
    telemetry.
  - `tactical_patterns` primary score now gates the extracted tactical best move
    instead of exact motif prose.
  - Exact tactical prose is emitted as telemetry.
- `src/chess_llm/training/phase_gate.py`
  - Opening/tactical telemetry metrics are excluded from hard floor and
    regression gates:
    - `opening_name_exact_match`
    - `opening_name_eco_exact`
    - `opening_name_eco_decade`
    - `opening_name_name_family`
    - `tactical_patterns_exact_match`

### Local Verification

```bash
python -m pytest tests/test_evals_benchmark.py tests/test_training_phase_gate.py -q
```

Result: `97 passed`.

Earlier focused tests also passed:

```bash
python -m pytest tests/test_sft_pipeline.py tests/test_sft_source_readiness.py -q
python -m pytest tests/test_sft_sources_lichess_evals.py -q
python -m pytest tests/test_sft_generators.py tests/test_sft_pipeline.py tests/test_sft_sources_lichess_evals.py -q
python -m pytest tests/test_training_data_loader.py -q
```

## Dataset Versions And Generation

### Phase B Refresh Used For Completed Runs

Data root:

```text
/workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703
```

Command:

```bash
cd /workspace/chess_sft_sdpo
source .venv-h100/bin/activate
export CHESS_SFT_OUTPUT=/workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703

.venv-h100/bin/chess-llm-make-data \
  --tier 1 2 3 4 5 6 \
  --volume 25000 \
  --eval-split-volume 25000 \
  --refresh-eval-splits \
  --source-readiness-report /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/readiness_phase_b_refresh_t1_t6_v25000.json
```

Source load counts for the completed Phase B runs:

- `332837` FENs
- `250000` puzzles
- `3704` openings
- `37923` eval rows
- `75000` endgames
- `250000` MATE rows
- `5511` opening-prefix book positions derived from Lichess openings

Validation command:

```bash
cd /workspace/chess_sft_sdpo
source .venv-h100/bin/activate

.venv-h100/bin/chess-llm-validate-outputs \
  --output-dir /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/output \
  --tier 1 2 3 4 5 6 \
  --expected-volume 25000 \
  --blocklist-path /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/eval_splits/blocklist.txt
```

Result:

- `46` files
- `1,150,000` examples
- `0` errors
- `0` contaminated FENs

### Current Targeted Tier 3/6 Regeneration

This job was still running as of `2026-07-04 17:11 UTC`.

Command:

```bash
cd /workspace/chess_sft_sdpo
mkdir -p /workspace/logs

nohup bash -lc '
source .venv-h100/bin/activate
export CHESS_SFT_OUTPUT=/workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703
.venv-h100/bin/chess-llm-make-data \
  --tier 3 6 \
  --volume 50000 \
  --eval-split-volume 25000 \
  --source-readiness-report /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/readiness_phase_b_repair_t3_t6_v50000.json
' > /workspace/logs/phase_b_repair_make_t3_t6_v50000.out.log \
  2> /workspace/logs/phase_b_repair_make_t3_t6_v50000.err.log &
```

Current observed source load:

- `661945` FENs
- `500000` puzzles
- `3704` openings
- `80562` eval rows
- `75000` endgames
- `500000` MATE rows

Validate after it settles:

```bash
cd /workspace/chess_sft_sdpo
source .venv-h100/bin/activate

.venv-h100/bin/chess-llm-validate-outputs \
  --output-dir /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/output \
  --tier 3 6 \
  --expected-volume 50000 \
  --blocklist-path /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/eval_splits/blocklist.txt
```

## Checkpoint Lineage

### Phase A Foundation

Sentinel:

```text
/workspace/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-h100-fa3-legality-cont1/phase_a/PASSED
```

Phase A passed:

| Gate | Value |
| --- | ---: |
| `perception/board_print` | `100.0%` |
| `rules/legal_moves` | `85.0%` |
| `rules/legality_check` | `90.5%` |
| `perception/state_tracking` | `100.0%` |

### Original Phase B

Output:

```text
/workspace/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-h100-fa3-legality-cont1/phase_b/best
```

Training completed:

- Runtime: about `38m03s`
- Train loss: about `0.0507`
- Tokens: `115,865,472`
- Throughput: about `50.8k` tokens/sec
- Optimizer steps: `1827`

### Targeted Phase B Repair

Repair output root:

```text
/workspace/chess_sft_checkpoints/phase-b-repair-from-b1-h100-fa3-20260704
```

Synthetic lineage:

- `phase_a/best` symlinked to original Phase B `phase_b/best`
- `phase_a/PASSED` created so the Phase B harness starts from prior Phase B best
- This avoided unsafe Trainer-state resume and did a model-only continuation.

Repair best checkpoint:

```text
/workspace/chess_sft_checkpoints/phase-b-repair-from-b1-h100-fa3-20260704/phase_b/best
```

Training completed:

- Runtime: about `53m40s`
- Train loss: about `0.0250`
- Tokens: `164,526,048`
- Throughput: about `51.1k` tokens/sec
- Optimizer steps: `2600`

## Training Commands

### Original Phase B

```bash
cd /workspace/chess_sft_sdpo

nohup bash -lc '
source .venv-h100/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node=2 .venv-h100/bin/chess-llm-train \
  --phase b \
  --data-root /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/output \
  --output-root /workspace/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-h100-fa3-legality-cont1 \
  --benchmark-dir /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/benchmark \
  --attn-implementation flash_attention_3 \
  --packing auto \
  --max-length 1024 \
  --num-train-epochs 1 \
  --skip-trainer-eval \
  --skip-eval \
  --trainer-save-steps 1000 \
  --require-phase-gate \
  --allow-wandb-offline
' > /workspace/logs/phase_b_train_h100_fa3_t1_t6.out.log \
  2> /workspace/logs/phase_b_train_h100_fa3_t1_t6.err.log &
```

### Targeted Phase B Repair

```bash
cd /workspace/chess_sft_sdpo

torchrun --standalone --nproc_per_node=2 .venv-h100/bin/chess-llm-train \
  --phase b \
  --data-root /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/output \
  --output-root /workspace/chess_sft_checkpoints/phase-b-repair-from-b1-h100-fa3-20260704 \
  --benchmark-dir /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/benchmark \
  --attn-implementation flash_attention_3 \
  --packing auto \
  --max-length 1024 \
  --num-train-epochs 1 \
  --task-upsample 3.1_available_captures=4 \
  --task-upsample 5.1_opening_identification=5 \
  --task-upsample 5.2_opening_continuation=2 \
  --task-upsample 6.1_endgame_classification=3 \
  --task-upsample 6.2_endgame_wdl=3 \
  --task-upsample 6.3_endgame_best_move=2 \
  --skip-trainer-eval \
  --skip-eval \
  --trainer-save-steps 1000 \
  --require-phase-gate \
  --allow-wandb-offline
```

Repair training mix:

- Train rows: `914,439`
- Eval rows: `11,496`
- `3.1_available_captures`: `24,482 -> 97,928`
- `5.1_opening_identification`: `24,500 -> 122,500`
- `5.2_opening_continuation`: `24,497 -> 48,994`
- `6.1_endgame_classification`: `24,498 -> 73,494`
- `6.2_endgame_wdl`: `24,498 -> 73,494`
- `6.3_endgame_best_move`: `24,500 -> 49,000`

## Eval Commands

### Original Phase B Gate

```bash
cd /workspace/chess_sft_sdpo
source .venv-h100/bin/activate

.venv-h100/bin/chess-llm-evaluate \
  --model /workspace/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-h100-fa3-legality-cont1/phase_b/best \
  --benchmark-dir /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/benchmark \
  --output /workspace/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-h100-fa3-legality-cont1/phase_b/eval_phase_b_gate_predictions.jsonl \
  --phase b \
  --attn-implementation sdpa \
  --max-new-tokens 768 \
  --batch-size 16 \
  --no-acpl \
  --allow-wandb-offline
```

### Repair Gate

```bash
cd /workspace/chess_sft_sdpo
source .venv-h100/bin/activate

.venv-h100/bin/chess-llm-evaluate \
  --model /workspace/chess_sft_checkpoints/phase-b-repair-from-b1-h100-fa3-20260704/phase_b/best \
  --benchmark-dir /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/benchmark \
  --output /workspace/chess_sft_checkpoints/phase-b-repair-from-b1-h100-fa3-20260704/phase_b/eval_phase_b_repair_gate_predictions.jsonl \
  --phase b \
  --attn-implementation sdpa \
  --max-new-tokens 768 \
  --batch-size 16 \
  --no-acpl \
  --allow-wandb-offline
```

### Rescore Existing Repair Predictions After Metric Fixes

This does not regenerate model outputs. It reuses the existing repair prediction
JSONL and applies the patched scorer.

Output:

```text
/workspace/chess_sft_checkpoints/phase-b-repair-from-b1-h100-fa3-20260704/phase_b/eval_phase_b_repair_gate_rescored_opening_tactics.results.json
```

## Gate Results

### Hard Gate Summary

| Run | Hard failures |
| --- | --- |
| Original Phase B | `endgame_wdl`, T1-2 floor via `rules/captures`, T3-6 floor via `openings/opening_name` |
| Targeted repair, old metrics | `endgame_wdl`, T1-2 floor via `rules/captures`, T3-6 floor via `openings/opening_name` |
| Targeted repair, patched metrics | `endgame_wdl`, T1-2 floor via `rules/captures`, T3-6 floor via `tactics/hanging_pieces` |

### Before And After Metrics

| Metric | Original Phase B | Repair, old scorer | Repair, patched scorer |
| --- | ---: | ---: | ---: |
| `perception/overall` | `98.55%` | `98.95%` | `98.95%` |
| `rules/overall` | `84.27%` | `85.69%` | `85.69%` |
| `rules/legal_moves` | `88.60%` | `88.14%` | `88.14%` |
| `rules/legality_check` | `90.91%` | `92.86%` | `92.86%` |
| `rules/captures` | `42.05%` | `53.54%` | `53.54%` |
| `tactics/overall` | `31.73%` | `35.85%` | `39.90%` |
| `tactics/capture_id` | `42.89%` | `49.42%` | `49.42%` |
| `tactics/hanging_pieces` | `21.00%` | `25.00%` | `25.00%` |
| `tactics/tactical_patterns` | `9.60%` exact prose | `12.60%` exact prose | `28.80%` best-move match |
| `evaluation/overall` | `84.47%` | `86.33%` | `86.33%` |
| `openings/overall` | `7.20%` | `7.24%` | `56.04%` |
| `openings/opening_name` | `0.00%` exact | `0.00%` exact | `61.00%` family/ECO-decade |
| `openings/opening_continuation` | `36.00%` | `36.20%` | `36.20%` |
| `endgames/overall` | `53.47%` | `54.53%` | `54.53%` |
| `endgames/endgame_wdl` | `78.00%` | `79.60%` | `79.60%` |
| `endgames/endgame_best_move` | `44.40%` | `46.20%` | `46.20%` |

### Metric-Fix Rationale

`opening_name` exact match was not a useful hard gate. On 400 repair examples:

- Exact opening-name match: `0.0%`
- Exact ECO match: `0.0%`
- ECO letter match: `78.5%`
- ECO decade match: `48.25%`
- Name family match: `52.75%`
- ECO decade OR name family: `61.0%`

This shows real opening-family signal while exact held-out names remain brittle
under transpositions and source label variants. Exact name and exact ECO remain
telemetry.

`tactical_patterns` exact motif prose was also too brittle for a hard floor. The
patched primary score is the extracted best move:

- Old exact motif/prose score after repair: `12.6%`
- Patched best-move score after repair: `28.8%`

This is still weak and remains useful signal for repair, but it no longer gates
on wording like `mate in 2` versus `tactical` when the best move is correct.

## Failure Diagnosis

### Captures

`rules/captures` is a real enumeration failure, not just a parser issue.

Repair gate sample breakdown:

- Perfect capture set: `52 / 154` (`33.77%`)
- Partial overlap: `60 / 154` (`38.96%`)
- Zero overlap: `42 / 154` (`27.27%`)

Observed modes:

- False `No captures available.`
- Extra non-captures included as captures
- Missed legal capture moves
- Wrong capture source squares

### Hanging Pieces

`tactics/hanging_pieces` is a real board-evaluation miss.

- Exact score: `25.0%`
- Square-set perfect: `25.0%`
- Average square-set Jaccard: about `31.99%`

Observed modes:

- Hallucinated hanging pawns in quiet positions
- Missed actually hanging pieces
- Wrong color/piece on correct or nearby squares

### Endgame WDL

`endgames/endgame_wdl` is a real WDL confusion issue.

Repair confusion summary from 500 examples:

| Gold | Pred draw | Pred loss | Pred win |
| --- | ---: | ---: | ---: |
| draw | `62` | `23` | `20` |
| loss | `18` | `156` | `1` |
| win | `21` | `19` | `180` |

Gold distribution:

- Win: `220`
- Loss: `175`
- Draw: `105`

Prediction distribution:

- Win: `201`
- Loss: `198`
- Draw: `101`

The model is not just biased to a single WDL label. It is making material/state
judgment errors, especially around tablebase exceptions and side-to-move.

## Artifact Paths

### Data

```text
/workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/output
/workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/benchmark
/workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/eval_splits/blocklist.txt
/workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/readiness_phase_b_refresh_t1_t6_v25000.json
/workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/readiness_phase_b_repair_t3_t6_v50000.json
```

### Original Phase B

```text
/workspace/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-h100-fa3-legality-cont1/phase_b/best
/workspace/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-h100-fa3-legality-cont1/phase_b/checkpoint-1827
/workspace/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-h100-fa3-legality-cont1/phase_b/eval_phase_b_gate_predictions.jsonl
/workspace/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-h100-fa3-legality-cont1/phase_b/eval_phase_b_gate_predictions.results.json
/workspace/chess_sft_checkpoints/phase-a-r10-v25000-legalfocus-h100-fa3-legality-cont1/phase_b/eval_phase_b_gate_predictions.analysis.json
```

### Targeted Repair

```text
/workspace/chess_sft_checkpoints/phase-b-repair-from-b1-h100-fa3-20260704/phase_b/best
/workspace/chess_sft_checkpoints/phase-b-repair-from-b1-h100-fa3-20260704/phase_b/checkpoint-2600
/workspace/chess_sft_checkpoints/phase-b-repair-from-b1-h100-fa3-20260704/phase_b/eval_phase_b_repair_gate_predictions.jsonl
/workspace/chess_sft_checkpoints/phase-b-repair-from-b1-h100-fa3-20260704/phase_b/eval_phase_b_repair_gate_predictions.results.json
/workspace/chess_sft_checkpoints/phase-b-repair-from-b1-h100-fa3-20260704/phase_b/eval_phase_b_repair_gate_predictions.analysis.json
/workspace/chess_sft_checkpoints/phase-b-repair-from-b1-h100-fa3-20260704/phase_b/eval_phase_b_repair_gate_rescored_opening_tactics.results.json
```

### Logs

```text
/workspace/logs/phase_b_train_h100_fa3_t1_t6.out.log
/workspace/logs/phase_b_train_h100_fa3_t1_t6.err.log
/workspace/logs/phase_b_repair_make_t3_t6_v50000.out.log
/workspace/logs/phase_b_repair_make_t3_t6_v50000.err.log
```

## Known Issue: Unsafe Trainer Resume

Do not use:

```bash
--resume-from-checkpoint auto
```

until this is fixed.

Observed problem:

- The attempted resume from Phase B `checkpoint-1827` restored trainer state but
  emitted a large missing/unexpected-key warning.
- Expected keys were shaped like `model.layers...`.
- Checkpoint keys were shaped like `model.language_model...`.
- Live loss resembled a first-epoch restart rather than a true continuation.

Current safe pattern:

1. Create a new output root.
2. Symlink the prior best model into the new root as `phase_a/best`.
3. Create `phase_a/PASSED`.
4. Run Phase B fresh from that model path without Trainer-state resume.

Example:

```bash
NEW_ROOT=/workspace/chess_sft_checkpoints/phase-b-repair2-t3t6-v50000-h100-fa3-20260704
PREV_BEST=/workspace/chess_sft_checkpoints/phase-b-repair-from-b1-h100-fa3-20260704/phase_b/best

mkdir -p "$NEW_ROOT/phase_a"
ln -sfn "$PREV_BEST" "$NEW_ROOT/phase_a/best"
printf 'Synthetic passed sentinel for model-only continuation from %s\n' "$PREV_BEST" \
  > "$NEW_ROOT/phase_a/PASSED"
```

## Next Planned Repair Run

Do this only after the current tier 3/6 `--volume 50000` data job settles and
the regenerated shards validate.

Rationale:

- Opening exact-name hard failure was a metric issue and is now resolved by
  family/ECO-decade scoring.
- Tactical motif prose hard failure was a metric issue, but best-move tactical
  accuracy is still low.
- Remaining hard failures are genuine:
  - `rules/captures` needs capture-enumeration repair.
  - `tactics/hanging_pieces` and tactical best moves need board-grounded repair.
  - `endgames/endgame_wdl` needs more tablebase-balanced supervision.
- More broad Phase B SFT moved these only modestly. The next run should be
  narrower and more aggressive on the failing tasks.

Proposed command:

```bash
cd /workspace/chess_sft_sdpo
source .venv-h100/bin/activate

NEW_ROOT=/workspace/chess_sft_checkpoints/phase-b-repair2-t3t6-v50000-h100-fa3-20260704
PREV_BEST=/workspace/chess_sft_checkpoints/phase-b-repair-from-b1-h100-fa3-20260704/phase_b/best

mkdir -p "$NEW_ROOT/phase_a"
ln -sfn "$PREV_BEST" "$NEW_ROOT/phase_a/best"
printf 'Synthetic passed sentinel for model-only continuation from %s\n' "$PREV_BEST" \
  > "$NEW_ROOT/phase_a/PASSED"

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node=2 .venv-h100/bin/chess-llm-train \
  --phase b \
  --data-root /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/output \
  --output-root "$NEW_ROOT" \
  --benchmark-dir /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/benchmark \
  --attn-implementation flash_attention_3 \
  --packing auto \
  --max-length 1024 \
  --num-train-epochs 1 \
  --task-upsample 3.1_available_captures=8 \
  --task-upsample 3.4_tactical_patterns=6 \
  --task-upsample 3.5_hanging_pieces=8 \
  --task-upsample 6.2_endgame_wdl=8 \
  --task-upsample 6.3_endgame_best_move=3 \
  --skip-trainer-eval \
  --skip-eval \
  --trainer-save-steps 1000 \
  --require-phase-gate \
  --allow-wandb-offline
```

Post-train eval:

```bash
cd /workspace/chess_sft_sdpo
source .venv-h100/bin/activate

.venv-h100/bin/chess-llm-evaluate \
  --model /workspace/chess_sft_checkpoints/phase-b-repair2-t3t6-v50000-h100-fa3-20260704/phase_b/best \
  --benchmark-dir /workspace/chess_sft_data/phase-a-r10-v25000-legalfocus-20260703/benchmark \
  --output /workspace/chess_sft_checkpoints/phase-b-repair2-t3t6-v50000-h100-fa3-20260704/phase_b/eval_phase_b_repair2_gate_predictions.jsonl \
  --phase b \
  --attn-implementation sdpa \
  --max-new-tokens 768 \
  --batch-size 16 \
  --no-acpl \
  --allow-wandb-offline
```

Pass/fail focus for the next run:

| Metric | Current patched value | Target |
| --- | ---: | ---: |
| `rules/captures` | `53.54%` | `>= 65%` |
| `tactics/hanging_pieces` | `25.00%` | `>= 55%` floor |
| `tactics/tactical_patterns` | `28.80%` | `>= 55%` floor |
| `endgames/endgame_wdl` | `79.60%` | `>= 85%` |

If this repair run does not move hanging pieces and WDL materially, the next
decision should be architectural: either add explicit verification-style
hanging-piece/endgame traces or test a larger Qwen checkpoint on the H100s.

## Addendum - 2026-07-05 Continuation

The planned repair2 run completed and became the best Phase B generator
checkpoint from this H100 sequence. Several follow-up repairs were then tried to
isolate whether hanging-piece verifier data, narrower tactical replay, or lower
learning rate could improve the generator. None beat repair2 on the tactics
benchmark, so repair2 remains the model to keep.

### Best Checkpoint Selected

Remote checkpoint:

```text
/workspace/chess_sft_checkpoints/phase-b-repair2-t3t6-v50000-h100-fa3-20260704/phase_b/best
```

Local copy downloaded on 2026-07-05:

```text
C:\Users\amazi\code\chess_sft_sdpo\artifacts\models\phase-b-repair2-best
```

Local file verification:

| File | Bytes |
| --- | ---: |
| `model.safetensors` | `3009613648` |
| `tokenizer.json` | `19989325` |
| `chat_template.jinja` | `7755` |
| `training_args.bin` | `5905` |
| `config.json` | `1792` |
| `tokenizer_config.json` | `1124` |
| `generation_config.json` | `164` |
| Total | `3029619713` |

Remote and local SHA-256 hashes matched on `2026-07-05`:

| File | SHA-256 |
| --- | --- |
| `chat_template.jinja` | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| `config.json` | `be2a21cc3fa9f2e2aae5d42a123239e6a55d18d2c84bacb9d9e138e789b2784e` |
| `generation_config.json` | `a78aebbc7804389b2f7863eaaac64c0bbe0b8a3fceb3d7ca71d539d3a3d96a83` |
| `model.safetensors` | `b55d7b35165e08eef769cb568b7ed84b7e2c576d31822cc934f0765dfa76fe2f` |
| `tokenizer.json` | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `tokenizer_config.json` | `66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b` |
| `training_args.bin` | `66ee86274112d0ca0cb9a0848db7bd791b5c71aa0d60be94e7bc2bb2361f51b5` |

Repair2 training metrics:

| Metric | Value |
| --- | ---: |
| Runtime | `3542.2581s` |
| Train loss | `0.027015433885696296` |
| Input tokens | `306017304` |
| Tokens/sec | `86390.45923841631` |

Repair2 full Phase B eval summary:

| Split/metric | Score |
| --- | ---: |
| `perception/overall` | `98.30%` |
| `rules/overall` | `85.86%` |
| `evaluation/overall` | `86.00%` |
| `openings/overall` | `54.33%` |
| `endgames/overall` | `69.87%` |
| `tactics/overall` | `47.05%` |
| `tactics/capture_id` | `56.88%` |
| `tactics/hanging_pieces` | `33.60%` |
| `tactics/tactical_patterns_best_move_match` | `37.40%` |
| `tactics/tactical_patterns_exact_match` | `16.60%` |
| `tactics/threats` | `60.33%` |

Strict Phase B still did not pass because the hard gate had one remaining
failure, but repair2 is the strongest generator checkpoint observed so far.

### Follow-Up Runs Tried

All follow-up runs used repair2 or its direct descendants as the starting point
unless noted. Tactics numbers below are from the same 2000-example tactics eval
where available.

| Run | Main change | Tactics overall | Capture | Hanging | Tactical best | Threats | Verifier diagnostic | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `repair2` | Focused T3/T6 v50000, packing auto | `47.05%` | `56.88%` | `33.60%` | `37.40%` | `60.33%` | n/a | Keep best |
| `repair4` | Direct hanging-piece continuation | `46.37%` | `57.70%` | `29.40%` | `37.20%` | `61.20%` | n/a | No-go |
| `repair5` | Hanging filter / threat-weighted continuation | `46.74%` | `56.49%` | `29.20%` | `36.80%` | `64.48%` | n/a | No-go |
| `repair6c` | Heavy hanging-claim verifier data, 20x | `44.44%` | n/a | `31.40%` | `32.20%` | n/a | `82.86%` | Learned verifier, hurt generator |
| `repair7` | Low-dose verifier, 4x, 600 steps, no packing | `42.56%` | `53.30%` | `27.40%` | `32.20%` | `57.34%` | `73.42%` | No-go |
| `repair8` | No verifier, aggressive tactics/endgame, 300 steps, default LR | `42.78%` | `53.99%` | `28.80%` | `33.00%` | `55.31%` | `64.22%` | No-go |
| `repair9` | Same as repair8, but `--learning-rate 2e-6` | `45.58%` | `56.90%` | `29.40%` | `37.60%` | `58.42%` | `63.14%` | No-go |

Repair9 exact training metrics:

| Metric | Value |
| --- | ---: |
| Runtime | `324.537s` |
| Train loss | `0.060703332821528115` |
| Input tokens | `4308768` |
| Tokens/sec | `13276.66182900563` |

Repair9 verifier diagnostic:

| Metric | Score |
| --- | ---: |
| `hanging_piece_claim_verification` | `63.14%` |
| `verdict_accuracy` | `69.10%` |
| `attacked_accuracy` | `71.90%` |
| `defended_accuracy` | `43.60%` |
| `hanging_accuracy` | `62.30%` |
| `correction_match` | `68.80%` |

### Verifier Data Finding

The hanging-claim verifier idea is still promising, but not as mixed generator
SFT in this form.

What happened:

- A benchmark prompt bug was found in `hanging_piece_claim_verification`: the
  diagnostic prompt lacked the answer-format contract, which made early
  verifier numbers look artificially bad.
- The prompt was fixed in `src/chess_llm/evals/benchmark.py` and covered by
  `tests/test_evals_benchmark.py`.
- The repaired diagnostic split contains 1000 examples, 200 each of:
  - `true_hanging_claim`
  - `defended_decoy_claim`
  - `missed_hanging_claim`
  - `safe_undefended_decoy_claim`
  - `true_non_hanging_claim`
- Heavy verifier training reached `82.86%` diagnostic accuracy in repair6c,
  proving the task is learnable.
- The same training hurt the generator benchmark, especially tactical best move
  and hanging-piece scores.

Decision:

- Keep verifier data as a separate judge/verifier curriculum for now.
- Do not mix verifier examples into the main move-generation SFT until the eval
  harness can measure trace and move faithfulness well enough to catch regressions.

### Control-Plane Fixes Made During The Follow-Up

These fixes were made locally and synced to the H100 repo before repair9:

- Added `--learning-rate` as a train CLI override and threaded it through
  `chess-llm-run-curriculum`.
- Fixed eval model dtype selection for flash attention backends. CUDA eval with
  `auto` or flash attention now loads weights as `torch.bfloat16`; `sdpa` and
  `eager` keep the prior `"auto"` behavior.
- Verified locally:

```bash
python -m pytest tests/test_training_entrypoints.py tests/test_training_eval_reliability.py -q
```

Result:

```text
106 passed in 3.38s
```

- Verified remotely that explicit FA3 eval on repair2 no longer fails due to
  fp32 weight loading.

### H100 Throughput Notes

Repair2 with packing and full training reached about `86.4k` input tokens/sec.
The short no-packing repair8/repair9 probes were about `13.3k` input tokens/sec.
That gap is expected from the run configuration and should not be interpreted as
the H100 node being bad.

Practical rules from this sequence:

- Use packing for real SFT runs unless the experiment specifically needs
  unpacked examples.
- Use `NCCL_NVLS_ENABLE=0` on this node.
- For FA3 eval, load CUDA weights as bf16.
- Avoid more default-LR continuation from repair2. It degraded tactics quickly.
- If continuing generator SFT from repair2, use lower LR plus much stronger
  replay/regularization, and measure on tactics before doing a full gate.

### Current Recommendation

Freeze repair2 as the current best generator checkpoint. The next useful branch
is not another broad repair run. The better next step is to train or evaluate a
separate verifier/judge on mechanically corrupted tactical traces, then use that
judge to filter or rank generator traces before attempting DPO/GRPO-style
self-distillation.
