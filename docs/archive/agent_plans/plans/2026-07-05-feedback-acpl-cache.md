# Feedback ACPL Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache scalar Stockfish evaluations used by ACPL in the existing evaluation SQLite sidecar.

**Architecture:** Extend `SqliteMultipvCache` with a `scalar_evaluations` table and methods for keyed scalar centipawn scores. Thread an optional `cache_path` through ACPL helpers in both evaluation entrypoints and reuse the same `*.multipv.sqlite` file as WPD.

**Tech Stack:** Python, sqlite3, python-chess, pytest.

---

### Task 1: Red Tests

**Files:**
- Modify: `tests/test_training_eval_reliability.py`
- Modify: `tests/test_evals_cli.py`

- [x] Add a training-eval ACPL test that proves a second run reuses cached position and post-move scores.
- [x] Add a standalone benchmark ACPL test that proves a second run reuses cached predicted post-move scores.
- [x] Run both tests and confirm RED.

### Task 2: Implementation

**Files:**
- Modify: `src/chess_llm/external/multipv.py`
- Modify: `src/chess_llm/training/evaluate.py`
- Modify: `src/chess_llm/evals/run_benchmark.py`

- [x] Add scalar cache get/put methods and table creation to `SqliteMultipvCache`.
- [x] Add cached ACPL evaluation wrappers in `chess_llm.training.evaluate`.
- [x] Add cached predicted-move evaluation in `chess_llm.evals.run_benchmark`.
- [x] Pass `*.multipv.sqlite` to ACPL and WPD in both eval entrypoints.

### Task 3: Verification And Commit

**Files:**
- All files above plus docs.

- [x] Run focused ACPL cache tests.
- [x] Run broader eval/cache tests.
- [x] Run the full test suite.
- [x] Inspect staged diff.
- [x] Commit the ACPL cache slice.
