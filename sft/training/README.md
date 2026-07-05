# Training Launch Scaffolding

This directory contains runtime launch helpers for the package-owned training
and evaluation CLIs. It no longer contains Python training modules.

Use imports from `chess_llm.training.*` and commands from `pyproject.toml`:

```bash
python -m pip install -e ".[data,train,eval]"
chess-llm-train --phase a --dry-run
chess-llm-evaluate --model chess_sft_checkpoints/phase_a/best --benchmark-dir chess_sft_data/benchmark --output predictions.jsonl --phase a
chess-llm-run-curriculum --start-phase a --end-phase c --wandb-project chess-sft --run-prefix full-curriculum
```

## Files

| File | Purpose |
| --- | --- |
| `run-wsl.ps1` | Sync the repo to WSL and run package commands in a Linux environment. |
| `run-docker.ps1` | Run package commands in the training Docker image. |
| `run-eval-docker.ps1` | Run eval commands in the eval Docker image or Compose service. |
| `posthoc_eval_then_phase.ps1` | Helper for posthoc eval and phase progression workflows. |
| `Dockerfile` | Linux GPU training image. |
| `Dockerfile.eval` | Eval image for direct benchmark generation/scoring. |
| `compose.yaml` | Compose services for trainer and evaluator containers. |
| `.env.example` | Local host path, GPU, cache, and credential template. |
| `requirements.txt` | Training image Python dependencies. |
| `requirements.eval.txt` | Eval image Python dependencies. |
| `build-causal-conv1d-wsl.sh` | WSL helper for source-building `causal-conv1d`. |

Unit tests live in the repo-root `tests/` directory.

## WSL Launcher

The WSL launcher reads `sft/training/.env` for persistent defaults and runtime
tokens. Put `HF_TOKEN` and `WANDB_API_KEY` there for Codex-managed launches, or
set them in PowerShell before launching. PowerShell environment variables override `.env` values.
You can also run `wandb login` inside the same WSL distro and user.
`WANDB_DISABLED` is forwarded so accidental host-side disables fail fast
instead of being hidden.

Verify the target environment:

```powershell
.\sft\training\run-wsl.ps1 -Distro Ubuntu-24.04-CUDA -- nvidia-smi
.\sft\training\run-wsl.ps1 -Distro Ubuntu-24.04-CUDA -- python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\sft\training\run-wsl.ps1 -Distro Ubuntu-24.04-CUDA -- chess-llm-train --phase a --dry-run
```

Use persistent Linux filesystem paths for real runs:

```powershell
.\sft\training\run-wsl.ps1 `
  -WslRepoPath /path/to/wsl/chess_sft_sdpo `
  -VenvPath /path/to/wsl/chess_sft_sdpo/.venv `
  -WslDataRoot /path/to/persistent/chess_sft_data `
  -WslCheckpointRoot /path/to/persistent/chess_sft_checkpoints `
  -WslHfCache /path/to/persistent/chess_sft_hf_cache `
  -WslWandbDir /path/to/persistent/chess_sft_wandb `
  -- chess-llm-train --phase a --dry-run
```

`-WslDataRoot` is the `CHESS_SFT_OUTPUT` root. It should contain generated
`output/`, `eval_splits/`, and `benchmark/` directories; do not point it at the `output/` directory itself.
Pass `-WslHfCache` to avoid repeated Qwen/Qwen3.5-0.8B downloads.

For generated-data Phase A runs, prefer one pass over unique data:

```powershell
.\sft\training\run-wsl.ps1 `
  -WslRepoPath /path/to/wsl/chess_sft_sdpo `
  -VenvPath /path/to/wsl/chess_sft_sdpo/.venv `
  -WslDataRoot /path/to/persistent/chess_sft_data `
  -WslCheckpointRoot /path/to/persistent/chess_sft_checkpoints `
  -- chess-llm-train --phase a `
  --num-train-epochs 1 `
  --skip-trainer-eval `
  --wandb-project chess-sft `
  --run-name phase-a-real
```

For real runs, unset `WANDB_MODE=offline`. Smoke runs allow offline W&B automatically.
Use `--allow-wandb-offline` only for an intentional offline rehearsal.

Use `--` before commands that take single-dash options so PowerShell stops
parsing the remaining arguments as launcher parameters:

```powershell
.\sft\training\run-wsl.ps1 -- python -c "import torch; print(torch.cuda.is_available())"
.\sft\training\run-wsl.ps1 -NoSync -- pytest tests -q
```

## Docker

Build from the repo root:

```powershell
docker compose --env-file .\sft\training\.env -f .\sft\training\compose.yaml build
```

Run a training smoke:

```powershell
.\sft\training\run-docker.ps1 chess-llm-train --phase a --smoke-run --wandb-project chess-sft --run-name docker-phase-a-smoke
```

Run direct eval through the evaluator service:

```powershell
docker compose --env-file .\sft\training\.env -f .\sft\training\compose.yaml run --rm evaluator chess-llm-evaluate --help

.\sft\training\run-eval-docker.ps1 -BuildImage -UseCompose chess-llm-evaluate `
  --model /data/chess_sft_checkpoints/phase_a/best `
  --benchmark-dir /data/chess_sft_data/benchmark `
  --output /data/chess_sft_checkpoints/phase_a/eval_predictions.jsonl `
  --phase a `
  --inference-backend vllm
```

The containers set `CHESS_SFT_OUTPUT=/data/chess_sft_data`,
`CHESS_SFT_CHECKPOINTS=/data/chess_sft_checkpoints`, and
`STOCKFISH_PATH=/usr/games/stockfish`.

The current Qwen3.5 fast-path dependency pins used by runtime files are
`flash-linear-attention[cuda]==0.5.1` and `causal-conv1d==1.6.2.post1`.

## Current Phase A Runbook

For real Phase A runs, use
[`docs/runbooks/phase_a_real_run.md`](../../docs/runbooks/phase_a_real_run.md).
The runbook covers the current data scale ladder, SDPA baseline, W&B
requirements, and sidecar eval shape.
