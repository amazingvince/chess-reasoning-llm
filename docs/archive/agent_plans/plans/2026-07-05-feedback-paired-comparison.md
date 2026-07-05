# Feedback Paired Comparison Implementation Plan

**Goal:** Add an offline paired-comparison command for two benchmark prediction JSONLs.

**Architecture:** A new `chess_llm.evals.compare_predictions` module reuses frozen benchmark loading, prediction JSONL loading, and existing primary scorers. The CLI prints compact summaries and optionally writes a JSON report.

**Tech Stack:** Python, pytest, existing `chess_llm.evals` benchmark primitives.

---

### Task 1: Red Tests

**Files:**
- Test: `tests/test_evals_compare_predictions.py`

- [x] Add a unit test for exact two-sided McNemar p-values.
- [x] Add a functional test that scores two prediction JSONLs on the same examples and checks paired counts.
- [x] Add a CLI test that prints a report and writes JSON output.
- [x] Run the focused test file and confirm RED from the missing module.

### Task 2: Implementation

**Files:**
- Add: `src/chess_llm/evals/compare_predictions.py`
- Modify: `src/chess_llm/evals/__init__.py`
- Modify: `pyproject.toml`

- [x] Implement exact McNemar p-value calculation without adding a new dependency.
- [x] Load frozen benchmark examples from a benchmark directory.
- [x] Load first prediction per example from both JSONLs.
- [x] Score predictions with existing benchmark scorers.
- [x] Emit overall, split, and task-level paired summaries.
- [x] Add a console script entrypoint.

### Task 3: Verification And Commit

**Files:**
- All files above.

- [x] Run focused paired comparison tests.
- [x] Run broader eval CLI tests.
- [x] Run the full test suite.
- [x] Inspect staged diff.
- [x] Commit the paired comparison slice.
