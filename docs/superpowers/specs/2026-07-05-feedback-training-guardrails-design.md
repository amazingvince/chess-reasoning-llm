# Feedback Training Guardrails Design

## Goal

Turn two known training harness landmines into hard preflight failures:

- Single-process multi-GPU training outside `torchrun`.
- Resume checkpoints whose model weight keys do not match the loaded model.

## Scope

This slice modifies `chess-llm-train` only. It does not change curriculum mixing, training data, checkpoint layout, or post-training eval policy.

## Design

Before training, `_training_launcher_preflight_error` checks CUDA visibility. If more than one CUDA device is visible and the process was not launched by `torchrun` (`WORLD_SIZE > 1` and `LOCAL_RANK` set), training returns `EVAL_INFRA_FAILURE_EXIT_CODE` with a command hint:

```powershell
torchrun --nproc_per_node=N -m chess_llm.training.train ...
```

When `--resume-from-checkpoint` resolves to a checkpoint directory, `_resume_checkpoint_model_key_error` compares the loaded model's `state_dict()` keys with checkpoint model weight keys before `SFTTrainer.train(...)` runs. The checkpoint keys are read from Hugging Face index files or safetensors metadata without loading tensor payloads.

If any keys are missing or unexpected, training fails with `EVAL_INFRA_FAILURE_EXIT_CODE` instead of allowing Transformers/Trainer to silently restart or partially load a mismatched checkpoint.

## Tests

Add focused tests in `tests/test_training_entrypoints.py`:

- multi-GPU without `torchrun` returns an error,
- multi-GPU under `torchrun` is allowed,
- mismatched checkpoint model keys return a hard error,
- matching checkpoint model keys are allowed.
