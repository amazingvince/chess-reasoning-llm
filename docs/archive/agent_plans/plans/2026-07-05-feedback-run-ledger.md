# Feedback Run Ledger And Artifact Mirror Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional local run-ledger and artifact-mirroring support for benchmark evaluations.

**Architecture:** A focused `chess_llm.artifacts.eval_runs` helper finalizes eval-run sidecars, appends JSONL ledger rows, and mirrors existing sidecars. `chess_llm.training.evaluate` calls that helper for all eval-run exits, while `chess_llm.training.train` only forwards paths to the eval subprocess.

**Tech Stack:** Python, pytest, existing artifact schemas and evaluation flow.

---

### Task 1: Red Tests

**Files:**
- Add: `tests/test_eval_run_artifacts.py`
- Modify: `tests/test_training_eval_reliability.py`
- Modify: `tests/test_training_entrypoints.py`

- [x] Test standalone finalization appends one ledger row and mirrors sidecar files.
- [x] Test mirrored `*.eval_run.json` includes final mirror metadata.
- [x] Test ledger append-only behavior without mirroring.
- [x] Test `run_evaluation` writes ledger rows and mirror files.
- [x] Test training eval command forwards ledger and mirror paths.
- [x] Run focused tests and confirm RED.

### Task 2: Implementation

**Files:**
- Add: `src/chess_llm/artifacts/eval_runs.py`
- Modify: `src/chess_llm/artifacts/__init__.py`
- Modify: `src/chess_llm/training/evaluate.py`
- Modify: `src/chess_llm/training/train.py`

- [x] Implement `finalize_evaluation_artifacts`.
- [x] Export the helper from `chess_llm.artifacts`.
- [x] Add `run_ledger` and `artifact_mirror_dir` to `EvaluationConfig`.
- [x] Add `--run-ledger` and `--artifact-mirror-dir` to `chess-llm-evaluate`.
- [x] Finalize eval-run artifacts after every eval-run sidecar write.
- [x] Add matching `chess-llm-train` flags and forward them to eval subprocesses.

### Task 3: Verification And Commit

**Files:**
- All files above plus docs.

- [x] Run focused ledger/mirror tests.
- [x] Run broader eval and training entrypoint tests.
- [x] Run the full test suite.
- [x] Inspect staged diff.
- [x] Commit the run ledger slice.
