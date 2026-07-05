# Phase C Move-Only Training Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class trainer support for task include/exclude filters and move-only target rewriting so Phase C move-only probes can run from the canonical data root.

**Architecture:** The data loader will accept a small immutable transform config. It will filter rows and rewrite selected task targets while reading raw JSONL, before metadata is stripped and cached. The mixer and training entrypoints will pass that config through without changing default curriculum behavior.

**Tech Stack:** Python, HuggingFace `datasets`, pytest, existing `chess_llm.training` CLI.

---

### Task 1: Loader Transform Tests

**Files:**
- Modify: `tests/test_training_data_loader.py`
- Modify: `src/chess_llm/training/data/loader.py`

- [x] **Step 1: Write failing tests**

Add tests that call `TrainingDataTransformConfig`, `_write_sanitized_jsonl(..., transform_config=...)`, and `_sanitized_jsonl_path(..., transform_config=...)`.

- [x] **Step 2: Run loader tests and confirm RED**

Run:

```bash
python -m pytest tests/test_training_data_loader.py -q
```

Expected: fail because `TrainingDataTransformConfig` and transform parameters do not exist.

- [x] **Step 3: Implement loader config and row transform**

Add `TrainingDataTransformConfig`, task filtering, move-only assistant rewrite, empty dataset handling, and transformed cache-key suffixes.

- [x] **Step 4: Run loader tests and confirm GREEN**

Run:

```bash
python -m pytest tests/test_training_data_loader.py -q
```

Expected: pass.

### Task 2: Mixer Transform Tests

**Files:**
- Modify: `tests/test_training_data_mixer.py`
- Modify: `src/chess_llm/training/data/mixer.py`

- [x] **Step 1: Write failing mixer test**

Add a test proving `build_phase_dataset(..., data_transform_config=...)` can include only a selected task and skip tiers that become empty.

- [x] **Step 2: Run mixer test and confirm RED**

Run:

```bash
python -m pytest tests/test_training_data_mixer.py::test_build_phase_dataset_applies_task_include_and_skips_empty_tiers -q
```

Expected: fail because the mixer has no `data_transform_config` argument.

- [x] **Step 3: Thread transform config through phase and schedule builders**

Pass config to `load_tier_data`, skip empty tiers when concatenating, and apply the same count logic in summary helpers.

- [x] **Step 4: Run mixer tests and confirm GREEN**

Run:

```bash
python -m pytest tests/test_training_data_mixer.py -q
```

Expected: pass.

### Task 3: CLI Forwarding Tests

**Files:**
- Modify: `tests/test_training_entrypoints.py`
- Modify: `src/chess_llm/training/train.py`
- Modify: `src/chess_llm/training/run_curriculum.py`

- [x] **Step 1: Write failing CLI tests**

Add tests proving `chess-llm-train` parses `--task-include`, `--task-exclude`, and `--move-only-task`; add a curriculum command forwarding test.

- [x] **Step 2: Run CLI tests and confirm RED**

Run:

```bash
python -m pytest tests/test_training_entrypoints.py::test_train_cli_accepts_data_transform_overrides -q
```

Expected: fail because flags do not exist.

- [x] **Step 3: Implement CLI args and forwarding**

Add parse args, `_build_data_transform_config`, pass config into dry-run/training/schedule paths, and forward flags from `run_curriculum`.

- [x] **Step 4: Run focused CLI tests and confirm GREEN**

Run:

```bash
python -m pytest tests/test_training_entrypoints.py -q
```

Expected: pass.

### Task 4: Verification And H100 Dry Run

**Files:**
- Modify: `docs/experiments/runs/2026-07-05_phase_c_contractfix.md`

- [x] **Step 1: Run local focused suite**

Run:

```bash
python -m pytest tests/test_training_data_loader.py tests/test_training_data_mixer.py tests/test_training_entrypoints.py -q
```

Expected: pass.

- [x] **Step 2: Sync changed source files to H100**

Copy modified `src/chess_llm/training` files to `/workspace/chess_sft_sdpo`.

- [x] **Step 3: Run H100 dry-run using canonical data**

Run `chess-llm-train --phase c --dry-run` with `--task-include 7.1_best_move_selection`, `--task-include 7.2_puzzle_solving`, and matching `--move-only-task` flags.

- [x] **Step 4: Document supported workflow**

Update the Phase C run note with the supported command and dry-run result.
