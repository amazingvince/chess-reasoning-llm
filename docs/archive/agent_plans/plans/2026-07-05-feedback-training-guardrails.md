# Feedback Training Guardrails Implementation Plan

**Goal:** Make unsafe training resume and multi-GPU launcher states fail before expensive or misleading training begins.

**Architecture:** Add small helper checks in `chess_llm.training.train` and call them before trainer construction/resume. Keep tests focused on pure helper behavior with fake torch/model objects.

**Tech Stack:** Python, pytest, existing training CLI.

---

### Task 1: Red Tests

**Files:**
- Modify: `tests/test_training_entrypoints.py`

- [x] Test multi-GPU visibility without `torchrun` returns an actionable error.
- [x] Test `torchrun` environment allows multi-GPU visibility.
- [x] Test mismatched resume checkpoint model keys return a hard error.
- [x] Test matching resume checkpoint model keys are allowed.
- [x] Run focused tests and confirm RED.

### Task 2: Implementation

**Files:**
- Modify: `src/chess_llm/training/train.py`

- [x] Add `_training_launcher_preflight_error`.
- [x] Call the launcher preflight before training starts.
- [x] Add checkpoint model-key inspection from HF index files or safetensors metadata.
- [x] Add `_resume_checkpoint_model_key_error`.
- [x] Call the resume key check before `SFTTrainer.train`.

### Task 3: Verification And Commit

**Files:**
- All files above.

- [x] Run focused guardrail tests.
- [x] Run broader training entrypoint tests.
- [x] Run the full test suite.
- [x] Inspect staged diff.
- [x] Commit the training guardrails slice.
