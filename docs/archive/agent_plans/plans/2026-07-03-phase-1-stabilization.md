# Phase 1 Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the current uncommitted branch before adding the reasoning-expansion work.

**Architecture:** This pass is hygiene and verification only. It removes accidental workspace baggage, records external-repo takeaways as prose, checks that docs and code describe the same current process, and runs the existing tests to expose regressions already present in the branch.

**Tech Stack:** PowerShell, Git, pytest, Markdown docs, existing `chess_llm` Python package.

---

### Task 1: Record External Generator Takeaways

**Files:**
- Create: `docs/reviews/2026-07-03_reasoning_dataset_generator_scan.md`

- [ ] **Step 1: Confirm the nested checkout is not a parent submodule**

Run: `git submodule status`
Expected: no entries.

- [ ] **Step 2: Confirm the nested checkout has been removed**

Run: `Test-Path .\reasoning_dataset_generator`
Expected: `False`.

- [ ] **Step 3: Write the scan note**

Create `docs/reviews/2026-07-03_reasoning_dataset_generator_scan.md` with:

```markdown
# reasoning_dataset_generator Scan Notes

Reviewed source: https://github.com/amazingvince/reasoning_dataset_generator.git at commit e58be9d153fec1c410cdba552fb0b5bd05e1ef9f.

## Keep As Ideas

- MultiPV-backed candidate tables and cached top-N move evaluations.
- Shallow-vs-deep trap/resource detection for candidate scan examples.
- Win Probability Delta scoring as a future reward/diagnostic shape.
- Per-prompt diversity telemetry: unique parsed moves, mode share, parse rate, reward standard deviation.
- Streaming JSONL sharding/upload ergonomics for long-running data generation.

## Do Not Import

- The free-form trace generator: it conflicts with the fixed-grammar, machine-verifiable direction in `plan/06_reasoning_expansion.md`.
- The standalone SFT/GRPO/UI pipeline: this repo already owns those surfaces.
- Direct code: the nested checkout had no top-level license file.

## Decision

Delete the nested checkout and keep this note as the only retained artifact.
```

- [ ] **Step 4: Verify the note is the only retained reference**

Run: `rg -n "reasoning_dataset_generator" . -g "!*.pyc"`
Expected: only intentional doc references, not a live source folder.

### Task 2: Audit Temporary And Untracked Artifacts

**Files:**
- Inspect: `docs/experiments/2026-07-03_tmp_run_results.md`
- Inspect: `docs/reviews/2026-07-03_deep_code_review.md`
- Inspect: `plan/06_reasoning_expansion.md`

- [ ] **Step 1: List untracked files**

Run: `git status --short`
Expected: docs and new source/test files are visible; `reasoning_dataset_generator/` is absent.

- [ ] **Step 2: Classify untracked docs**

Keep these as intentional branch artifacts:
- `docs/experiments/2026-07-03_tmp_run_results.md`
- `docs/reviews/2026-07-03_deep_code_review.md`
- `docs/reviews/2026-07-03_reasoning_dataset_generator_scan.md`
- `plan/06_reasoning_expansion.md`

- [ ] **Step 3: Classify untracked source/tests**

Keep these as intentional code additions for the current branch:
- `src/chess_llm/autodata/selfplay.py`
- `src/chess_llm/core/rays.py`
- `src/chess_llm/sft/sources/self_play.py`
- `src/chess_llm/training/schedule.py`
- `src/chess_llm/training/schedule_callback.py`
- `tests/test_autodata_selfplay.py`
- `tests/test_sft_sources_self_play.py`
- `tests/test_training_schedule.py`

### Task 3: Run Stabilization Verification

**Files:**
- Test command covers the full repository.

- [ ] **Step 1: Run full pytest**

Run:

```powershell
python -m pytest -q -p no:cacheprovider --basetemp .tmp\pytest-phase1-full
```

Expected: all tests pass, or failures are documented and fixed in a follow-up task.

- [ ] **Step 2: Run diff whitespace check**

Run:

```powershell
git diff --check
```

Expected: exit code 0. Git line-ending normalization warnings may appear; whitespace errors must be fixed.

### Task 4: Fix Any Stabilization Regressions

**Files:**
- Modify only files implicated by failing tests or whitespace errors.

- [ ] **Step 1: For each test failure, write or use the failing test as the reproducer**

Run the smallest failing pytest node with:

```powershell
python -m pytest path\to\test.py::test_name -q -p no:cacheprovider --basetemp .tmp\pytest-phase1-one
```

Expected: failure reproduces before code changes.

- [ ] **Step 2: Apply the minimal fix**

Use `apply_patch` only. Keep fixes scoped to the failing behavior.

- [ ] **Step 3: Verify the focused test passes**

Run the same smallest pytest node.
Expected: pass.

- [ ] **Step 4: Rerun full pytest and diff check**

Run:

```powershell
python -m pytest -q -p no:cacheprovider --basetemp .tmp\pytest-phase1-full-final
git diff --check
```

Expected: pytest passes; diff check exits 0.
