# Feedback Eval Preflight Implementation Plan

**Goal:** Add early eval-harness checks for decode budget, required Stockfish, and optional benchmark manifest identity.

**Architecture:** `chess_llm.training.evaluate` computes selected benchmark examples and required ACPL splits, then runs `_eval_preflight_errors` before W&B or model loading. Failures produce the normal eval-run artifact with an infra failure return code.

**Tech Stack:** Python, pytest, existing `chess_llm.training.evaluate` evaluation flow.

---

### Task 1: Red Tests

**Files:**
- Modify: `tests/test_training_eval_reliability.py`

- [x] Test Phase C planning traces reject too-small `max_new_tokens` before model load.
- [x] Test Phase C ACPL rejects missing Stockfish before model load.
- [x] Test explicit manifest enforcement rejects missing `manifest.json`.
- [x] Run focused tests and confirm RED.

### Task 2: Implementation

**Files:**
- Modify: `src/chess_llm/training/evaluate.py`

- [x] Add strict preflight knobs to `EvaluationConfig` and CLI parsing.
- [x] Add side-effect-free Stockfish path resolution.
- [x] Add `_eval_preflight_errors` for decode budget, Stockfish, and manifest checks.
- [x] Run preflight after benchmark loading and before W&B/model loading.
- [x] Write infra-failure eval-run artifacts for preflight failures.

### Task 3: Verification And Commit

**Files:**
- All files above.

- [x] Run focused preflight tests.
- [x] Run broader training eval reliability tests.
- [x] Run the full test suite.
- [x] Inspect staged diff.
- [x] Commit the eval preflight slice.
