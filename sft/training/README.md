# training - Chess SFT Training Harness

Three-phase supervised fine-tuning harness for the chess SFT curriculum in `sft/make_data/output/`.

This folder is now a compatibility layer. The implementation lives under
`src/chess_llm/training/`; `train.py`, `evaluate.py`, `run_curriculum.py`, and
`config/training_args.py` forward into the package modules.

This package handles:
- phase-aware dataset mixing across tiers 1-7
- chained A -> B -> C training
- benchmark evaluation and phase gating
- bounded smoke runs for end-to-end verification
- optional `transformers` or `vllm` inference for eval

Default base model: `Qwen/Qwen3.5-0.8B`

Override it with:

```bash
chess-llm-train --phase a --base-model Qwen/Qwen3.5-0.8B
```

## Quick Start

Prefer package CLIs from the repo root after installing the package:

```bash
python -m pip install -e ".[data,train,eval]"
chess-llm-train --phase a --dry-run
chess-llm-train --phase a --smoke-run --wandb-project chess-sft --run-name phase-a-smoke
chess-llm-run-curriculum --wandb-project chess-sft --run-prefix curriculum
```

Legacy wrapper files still work for backward compatibility, but new workflows
should use the package entry points.

```bash
# Install data-generation, training, and evaluation dependencies
python -m pip install -e ".[data,train,eval]"

# Inspect phase config and expected data counts
chess-llm-train --phase a --dry-run

# Short end-to-end smoke test
chess-llm-train --phase a --smoke-run --wandb-project chess-sft --run-name phase-a-smoke

# Full Phase A training
chess-llm-train --phase a --wandb-project chess-sft --wandb-group phase-a --run-name phase-a-full

# Full Phase B training (uses Phase A best/ checkpoint if present)
chess-llm-train --phase b --wandb-project chess-sft --wandb-group phase-b --run-name phase-b-full

# Full Phase C training
chess-llm-train --phase c --wandb-project chess-sft --wandb-group phase-c --run-name phase-c-full

# Full end-to-end curriculum: train -> post-eval telemetry across A/B/C
chess-llm-run-curriculum --wandb-project chess-sft --run-prefix full-curriculum
```

## What Lives Here

```text
training/
|-- config/
|   |-- phases.py         # Phase configs, tier mixes, checkpoint resolution
|   |-- training_args.py  # TRL SFTConfig builder
|-- data/
|   |-- loader.py         # Load tier JSONL into HuggingFace Dataset
|   |-- mixer.py          # FEN-safe train/eval split + per-phase mixing
|-- train.py              # Compatibility wrapper for chess_llm.training.train
|-- evaluate.py           # Compatibility wrapper for chess_llm.training.evaluate
|-- run_curriculum.py     # Compatibility wrapper for chess_llm.training.run_curriculum
|-- phase_gate.py         # Phase pass/fail logic
|-- Dockerfile            # Linux GPU training image
|-- compose.yaml          # Docker Compose service pinned to one GPU
|-- .env.example          # Host path and GPU selection template
|-- tests/                # Unit tests for mixing, gating, helpers
|-- requirements.txt
`-- README.md
```

## Data and Checkpoints

Default paths come from environment variables when present:

- `CHESS_SFT_OUTPUT` -> data root and benchmark root
- `CHESS_SFT_CHECKPOINTS` -> checkpoint root
- `CHESS_SFT_BASE_MODEL` -> base HF model ID for Phase A
- `STOCKFISH_PATH` -> Stockfish binary for ACPL

Default fallback paths in code are:

```text
data root:       chess_sft_data/output
benchmark root:  chess_sft_data/benchmark
checkpoints:     chess_sft_checkpoints
```

Phase outputs are written under:

```text
<output-root>/
|-- phase_a/
|   |-- best/
|   |-- eval_predictions.jsonl
|   |-- eval_predictions.analysis.json
|   |-- eval_predictions.eval_run.json
|   |-- eval_predictions.results.json
|   |-- READY
|   `-- PASSED        # only when --require-phase-gate is used and passes
|-- phase_b/
`-- phase_c/
```

By default, phase chaining is training-first: a later phase only needs the prior phase's `best/` checkpoint. `READY` marks a checkpoint that is available for continued training. `PASSED` is written only in explicit hard-gate mode.

Use `--require-phase-gate` when you want the older strict behavior where benchmark failures block progression.

## Phase Schedule

The curriculum is defined in [phases.py](/c:/Users/amazi/code/chess_sft_sdpo/sft/training/config/phases.py).

| Phase | Purpose | Tier Mix | LR |
|------|---------|----------|----|
| A | Foundation | T1-2 at 100% | `2e-5` |
| B | Understanding | T3-6 at 100% + T1-2 review at 30% | `1e-5` |
| C | Planning | T3-7 at 100% + T1-2 review at 20% + T7 upsampled 5x | `5e-6` |

## Core Commands

### Dry Run

Prints the phase summary without training:

```bash
chess-llm-train --phase a --dry-run
chess-llm-train --phase c --dry-run --inference-backend vllm
```

This is useful for checking:
- data availability
- tier counts
- phase LR / epochs
- smoke-run overrides

### Smoke Run

`--smoke-run` is a bounded end-to-end run for verifying the harness before a real job.

Current defaults:
- train examples: `128`
- trainer eval examples: `64`
- benchmark examples per split: `32`
- trainer `max_steps`: `10`

Example:

```bash
chess-llm-train --phase a --smoke-run --wandb-project chess-sft --run-name phase-a-smoke
```

You can also override the bounds directly:

```bash
chess-llm-train --phase a \
  --max-train-examples 256 \
  --max-eval-examples 64 \
  --max-benchmark-examples-per-split 50 \
  --max-steps 20 \
  --wandb-project chess-sft \
  --run-name phase-a-bounded-smoke
```

### Trainer Eval Defaults

Full runs use a bounded trainer-eval subset for loss tracking and best-checkpoint selection:
- trainer eval examples: `2048` by default
- trainer eval/save cadence: every `5000` optimizer steps
- benchmark eval remains telemetry by default
- bounded runs with `--skip-trainer-eval` do not add mid-run checkpoint saves
  unless `--trainer-save-steps <N>` is passed

Useful controls:
- `--max-eval-examples 0` to use the full mixed eval set during training
- `--trainer-eval-steps 2000 --trainer-save-steps 2000` for more frequent best-checkpoint checks
- `--skip-trainer-eval` to train without in-loop eval; pair it with
  `--trainer-save-steps <N>` when you still want periodic recoverable
  checkpoints plus the final `best/` export
- `--gradient-checkpointing` to trade throughput for lower activation memory
  when a larger model or batch no longer fits

### Training Speed Defaults

The default Qwen/Qwen3.5 training path keeps the optimizer loop lean:

- bf16 and CUDA TF32 are enabled
- TRL tokenization uses `dataset_num_proc`
- trainer eval is optional and defaults to a 2048-example cap
- Liger remains enabled, but Qwen3.5 `fused_linear_cross_entropy` is off by
  default because local Blackwell probes were slower with it enabled
- packing/padding-free batches are enabled only when the selected attention
  backend is FlashAttention 2, 3, or 4; under SDPA they stay off to avoid
  cross-example attention risk

Use `--enable-liger-fused-linear-ce` only as an explicit benchmark experiment.
Use `--disable-liger-kernel` when comparing against stock Transformers kernels.

### Eval Only

Run benchmark evaluation without training:

```bash
chess-llm-train --phase a --eval-only
chess-llm-train --phase c --eval-only --inference-backend vllm
```

Behavior:
- if `phase_<name>/best/` exists, it evaluates that checkpoint
- otherwise it falls back to the phase starting model

Evaluation runs now support W&B logging too. Use `--no-wandb` to disable it.

### Full Curriculum Runner

Run train and post-train benchmark telemetry for each phase:

```bash
chess-llm-run-curriculum --wandb-project chess-sft --run-prefix full-curriculum
```

Useful options:
- `--start-phase b --end-phase c` to resume from a later stage
- `--pre-eval` to add a lightweight report-only eval before each phase
- `--require-phase-gate` to make benchmark failures block progression
- `--smoke-run` to do a bounded end-to-end rehearsal
- `--run-prefix march19` to stamp consistent run names

The runner performs:
- training for that phase
- post-train benchmark eval via `chess-llm-train`
- automatic stop on training/runtime failures; benchmark metric failures stop only with `--require-phase-gate`

### Direct Benchmark Eval

You can also call `chess-llm-evaluate` directly:

```bash
chess-llm-evaluate \
  --model chess_sft_checkpoints/phase_a/best \
  --benchmark-dir chess_sft_data/benchmark \
  --output phase_a_predictions.jsonl \
  --phase a
```

Phase C with pass@8:

```bash
chess-llm-evaluate \
  --model chess_sft_checkpoints/phase_c/best \
  --benchmark-dir chess_sft_data/benchmark \
  --output phase_c_predictions.jsonl \
  --phase c \
  --pass-k 8
```

## Docker Training

You can run the harness inside Linux GPU containers managed from the repo root.

Files:
- [Dockerfile](/c:/Users/amazi/code/chess_sft_sdpo/sft/training/Dockerfile)
- [compose.yaml](/c:/Users/amazi/code/chess_sft_sdpo/sft/training/compose.yaml)
- [.env.example](/c:/Users/amazi/code/chess_sft_sdpo/sft/training/.env.example)
- [run-docker.ps1](/c:/Users/amazi/code/chess_sft_sdpo/sft/training/run-docker.ps1)

The compose setup defaults to:
- `CUDA_VISIBLE_DEVICES=0`
- eval `CUDA_VISIBLE_DEVICES=0`
- data root mounted from `../../.runtime/chess_sft_data`
- checkpoints mounted from `../../.runtime/chess_sft_checkpoints`
- HuggingFace cache mounted from `../../.runtime/hf_cache`
- `BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`
- `EVAL_BASE_IMAGE=vllm/vllm-openai:latest`
- `transformers>=5.3.0,<5.13.0`
- `INSTALL_VLLM=0`
- `INSTALL_FLASH_ATTN=1`

Set `CUDA_DEVICE_ID` and `EVAL_CUDA_DEVICE_ID` in `.env` when you want trainer
and evaluator containers to use different GPUs.

### First-Time Setup

1. Start Docker Desktop.
2. Make sure Docker Desktop is using Linux containers with the WSL2 backend.
3. Copy `.env.example` to `.env` in `sft/training/` and adjust paths if needed.

```powershell
Copy-Item .\sft\training\.env.example .\sft\training\.env
```

### Build the Image

From the repo root:

```powershell
docker compose --env-file .\sft\training\.env -f .\sft\training\compose.yaml build
```

If you want `vllm` installed in the image too:

```powershell
$env:INSTALL_VLLM="1"
docker compose --env-file .\sft\training\.env -f .\sft\training\compose.yaml build
```

Flash Attention 2 is installed from the matching published wheel by default on this machine:

```powershell
$env:INSTALL_FLASH_ATTN="1"
docker compose --env-file .\sft\training\.env -f .\sft\training\compose.yaml build
```

The default training image uses the exact wheel-backed combo that was verified locally:
- base image: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`
- Python: `3.11`
- Torch: `2.8.0+cu128`
- Qwen3.5 fast path: `flash-linear-attention[cuda]==0.5.1`, `causal-conv1d==1.6.2.post1`
- FlashAttention wheel: `flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl`

Qwen3.5's FLA fast path is separate from FlashAttention 2. FlashAttention 2
controls `attn_implementation="flash_attention_2"` when available; the Qwen3.5
Gated DeltaNet fast path needs `flash-linear-attention` plus `causal-conv1d`.
If those are missing, Transformers can still run through the torch fallback, but
model loading prints a warning.

### Recommended Way to Run on This Machine

Use the PowerShell helper:

```bash
.\sft\training\run-docker.ps1 chess-llm-train --phase a --smoke-run --wandb-project chess-sft --run-name docker-phase-a-smoke
```

Full end-to-end curriculum in Docker:

```bash
.\sft\training\run-docker.ps1 chess-llm-run-curriculum --wandb-project chess-sft --run-prefix docker-curriculum
```

That helper:
- mounts the repo and data directories
- requests GPU access from Docker
- sets `CUDA_VISIBLE_DEVICES` from `.env`

### Verify GPU Access

Open a shell in the container:

```bash
.\sft\training\run-docker.ps1
```

Then inside the container:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Run a Smoke Test in Docker

```bash
.\sft\training\run-docker.ps1 chess-llm-train --phase a --smoke-run --wandb-project chess-sft --run-name docker-phase-a-smoke
```

### Run Full Training

```bash
.\sft\training\run-docker.ps1 chess-llm-train --phase a --wandb-project chess-sft --wandb-group phase-a --run-name docker-phase-a-full
.\sft\training\run-docker.ps1 chess-llm-train --phase b --wandb-project chess-sft --wandb-group phase-b --run-name docker-phase-b-full
.\sft\training\run-docker.ps1 chess-llm-train --phase c --wandb-project chess-sft --wandb-group phase-c --run-name docker-phase-c-full
```

### Eval in Docker

Transformers backend:

```bash
.\sft\training\run-docker.ps1 chess-llm-evaluate \
  --model /data/chess_sft_checkpoints/phase_a/best \
  --benchmark-dir /data/chess_sft_data/benchmark \
  --output /data/chess_sft_checkpoints/phase_a/eval_predictions.jsonl \
  --phase a \
  --inference-backend transformers
```

vLLM backend:

```bash
docker compose --env-file .\sft\training\.env -f .\sft\training\compose.yaml run --rm evaluator \
  chess-llm-evaluate \
  --model /data/chess_sft_checkpoints/phase_a/best \
  --benchmark-dir /data/chess_sft_data/benchmark \
  --output /data/chess_sft_checkpoints/phase_a/eval_predictions.jsonl \
  --phase a \
  --inference-backend vllm
```

The PowerShell eval helper can build and launch that same Compose evaluator
service:

```powershell
.\sft\training\run-eval-docker.ps1 -BuildImage -UseCompose chess-llm-evaluate `
  --model /data/chess_sft_checkpoints/phase_a/best `
  --benchmark-dir /data/chess_sft_data/benchmark `
  --output /data/chess_sft_checkpoints/phase_a/eval_predictions.jsonl `
  --phase a `
  --inference-backend vllm
```

### Notes

- The container installs Linux `stockfish` and sets `STOCKFISH_PATH=/usr/games/stockfish`.
- Training still uses TRL / transformers.
- `vllm` is optional and off by default.
- Change `CUDA_DEVICE_ID` in `.env` if you want to target a different CUDA device later.
- Docker bind mounts from Windows paths work, but for best filesystem performance you may eventually want the repo and caches inside the WSL/Linux filesystem.

## WSL Training

The PowerShell launcher [run-wsl.ps1](/c:/Users/amazi/code/chess_sft_sdpo/sft/training/run-wsl.ps1) runs training from a Linux-side source copy while keeping large artifacts in configurable WSL paths.

Default WSL launcher layout:

```text
WSL distro:       Windows default WSL distro, or pass -Distro explicitly
WSL repo:         /tmp/chess_sft_sdpo
Conda env:        chess-sft-wsl, or a repo-local .venv via -VenvPath
Data root:        /tmp/chess_sft_data
Checkpoints:      /tmp/chess_sft_checkpoints
HF cache:         /tmp/chess_sft_hf_cache
W&B cache:        /tmp/chess_sft_wandb
Stockfish:        stockfish
PyTorch GPU id:   CUDA_VISIBLE_DEVICES=0
```

For real WSL training, pass a persistent cache such as
`-WslHfCache /path/to/persistent/chess_sft_hf_cache` to avoid repeated Qwen/Qwen3.5-0.8B downloads from the launcher's default `/tmp` cache.

Before a real Codex-managed run, verify the same launcher path you plan to use:
`rsync`, virtualenv or conda activation, `chess-llm-train --help`, `nvidia-smi`,
Torch CUDA visibility, Stockfish, and W&B auth.

The WSL launcher reads `sft/training/.env` for persistent defaults and runtime
tokens. Put `HF_TOKEN` and `WANDB_API_KEY` there for Codex-managed launches,
or set them in PowerShell before launching; PowerShell environment variables override `.env` values.
You can also run `wandb login` inside the same WSL distro and user.
`WANDB_DISABLED` is forwarded so accidental host-side disables fail fast instead
of being hidden.

For real runs, use persistent WSL paths instead of the launcher's `/tmp`
defaults:

```powershell
.\sft\training\run-wsl.ps1 `
  -WslDataRoot /path/to/persistent/chess_sft_data `
  -WslCheckpointRoot /path/to/persistent/chess_sft_checkpoints `
  -WslHfCache /path/to/persistent/chess_sft_hf_cache `
  -WslWandbDir /path/to/persistent/chess_sft_wandb `
  chess-llm-train --phase a --dry-run
```

`-WslDataRoot` should contain the generated `output/` and `benchmark/` directories used by training and posthoc evaluation.

For the current WSL venv, install the Qwen3.5 fast-path dependencies with:

```bash
VENV=/path/to/chess_sft_sdpo/.venv
$VENV/bin/python -m pip install wheel packaging ninja
$VENV/bin/python -m pip install --no-build-isolation "causal-conv1d==1.6.2.post1"
$VENV/bin/python -m pip install "flash-linear-attention[cuda]==0.5.1"
```

Use `--no-build-isolation` for `causal-conv1d`; otherwise pip may try to pull a
separate torch/CUDA build stack instead of compiling against the already-installed
torch. Verify with:

```bash
$VENV/bin/python -c "import fla, causal_conv1d; print('qwen fast path deps ok')"
```

Install FlashAttention separately from the Qwen3.5 Gated DeltaNet fast path:

```bash
VENV=/path/to/chess_sft_sdpo/.venv
$VENV/bin/python -m pip install wheel packaging ninja
$VENV/bin/python -m pip install "kernels>=0.14.0,<0.15.0"
$VENV/bin/python -m pip install --pre --index-url https://pypi.org/simple "flash-attn-4[cu13]"
```

The loader tries stable FlashAttention backends, then SDPA/eager in auto mode.
On the current CUDA 13 WSL stack, the verified padding-free/packing path is the
pinned Hugging Face `kernels-community/flash-attn2` kernel, but short Phase A
examples benchmark faster through SDPA than through the packed HF FA2 route. In
`auto` mode the loader tries:

1. FA4 only when `CHESS_SFT_ENABLE_FLASH_ATTENTION_4_AUTO=1`.
2. FA3 on Hopper.
3. `kernels-community/flash-attn2@bcc70a66fbe0445d4484f56167d706134d53633d`
   only when `CHESS_SFT_ENABLE_HF_FLASH_ATTN2_AUTO=1`.
4. locally installed `flash_attention_2`, then SDPA/eager.

Set this only when you explicitly want TRL packing/padding-free experiments:

```bash
export CHESS_SFT_ENABLE_HF_FLASH_ATTN2_AUTO=1
```

The older disable switch still overrides the opt-in if both are set:

```bash
export CHESS_SFT_DISABLE_HF_FLASH_ATTN2_AUTO=1
```

`flash-attn-4[cu13]` is still useful for explicit experiments; it is published
as a prerelease, so `--pre` is required. Keep FA4 out of auto mode for real runs
unless an explicit model-load smoke passes:

```bash
export CHESS_SFT_ENABLE_FLASH_ATTENTION_4_AUTO=1
```

Explicit `--attn-implementation flash_attention_4` remains available for
experiments, but SDPA is the safe default fallback on the local 5090 until FA4
is stable end to end.

For one source-built `causal-conv1d` install that can run the native 1D-conv
kernel on the local Ada/Blackwell pair and on Hopper, build it as a CUDA fat
binary:

```bash
VENV=/path/to/wsl/chess_sft_sdpo/.venv \
CUDA_HOME=/usr/local/cuda-12.8 \
TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0" \
bash /mnt/c/Users/amazi/code/chess_sft_sdpo/sft/training/build-causal-conv1d-wsl.sh
```

The current Qwen3.5 loader smoke-tests the optional 1D-conv kernels before using
them. If the installed wheel lacks the visible GPU architecture, inference and
eval continue on the Transformers fallback; only the Qwen3.5 fast path is
disabled. Blackwell `sm_120` needs a CUDA toolkit whose `nvcc` supports
`compute_120`; CUDA 12.3 only builds through `compute_90`.

NVIDIA cuDNN frontend also exposes `cudnn.ops.causal_conv1d`, but it needs a
new enough cuDNN runtime for the `cudnnCausalConv1dForward` symbol. Local probes
passed on both the 4090 and 5090 with:

```bash
$VENV/bin/python -m pip install "nvidia-cudnn-frontend==1.25.0" \
  "nvidia-cudnn-cu13==9.23.2.1" \
  "nvidia-cudnn-jit-cu13==9.23.2.1"
```

This is not a full Qwen3.5 fast-path replacement because Qwen3.5 also uses
`causal_conv1d_update` for incremental state updates; keep `causal-conv1d` as
the production dependency unless the model code is adapted.

Check PyTorch's CUDA device order before choosing `CUDA_VISIBLE_DEVICES`; it can
differ from `nvidia-smi`:

```bash
$VENV/bin/python -c "import torch; [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

Run commands from the Windows repo root:

```powershell
.\sft\training\run-wsl.ps1 chess-llm-train --phase a --dry-run
.\sft\training\run-wsl.ps1 chess-llm-train --phase a --smoke-run --wandb-project chess-sft --run-name wsl-phase-a-smoke
.\sft\training\run-wsl.ps1 chess-llm-train --phase a --wandb-project chess-sft --wandb-group phase-a --run-name wsl-phase-a-full
```

Use `--` before commands that take single-dash options so PowerShell stops
parsing the remaining arguments as launcher parameters:

```powershell
.\sft\training\run-wsl.ps1 -- python -c "import torch; print(torch.cuda.is_available())"
.\sft\training\run-wsl.ps1 -NoSync -- pytest tests -q
```

Use `-VenvPath` when the WSL repo has a local virtualenv instead of a conda
environment:

```powershell
.\sft\training\run-wsl.ps1 `
  -WslRepoPath /path/to/wsl/chess_sft_sdpo `
  -VenvPath /path/to/wsl/chess_sft_sdpo/.venv `
  chess-llm-train --phase a --dry-run
```

For the current Phase A square-mapping curriculum, run without
`--require-phase-gate` so benchmark failures are telemetry rather than a hard
block. The state-tracking and FEN-assembly tasks should be upsampled until UCI
square lookup/edit mechanics become reliable:

```powershell
.\sft\training\run-wsl.ps1 `
  -WslRepoPath /path/to/wsl/chess_sft_sdpo `
  -VenvPath /path/to/wsl/chess_sft_sdpo/.venv `
  -WslDataRoot /path/to/wsl/chess_sft_data_phase_a `
  -WslCheckpointRoot /path/to/wsl/chess_sft_checkpoints_phase_a `
  -WslHfCache /path/to/persistent/chess_sft_hf_cache `
  chess-llm-train --phase a `
  --task-upsample 1.5_state_tracking=4 `
  --task-upsample 1.9_fen_assembly=4 `
  --task-upsample 1.10_fen_row_application=4 `
  --skip-trainer-eval `
  --trainer-save-steps 1000 `
  --max-benchmark-examples-per-split 100 `
  --eval-batch-size 16 `
  --eval-max-new-tokens 192 `
  --no-acpl `
  --wandb-project chess-sft `
  --wandb-group phase-a-fenassembly `
  --run-name phase-a-fenassembly
```

For a bounded rehearsal, add `--max-steps 100 --max-train-examples 8192`.
For a real run, omit those caps and let the phase schedule run its configured
epochs. A real run expects online W&B by default; unset `WANDB_MODE=offline`
before launching. Add `--allow-wandb-offline` only for an intentional offline
W&B rehearsal, and add `--no-wandb` only when local-only logging is desired.

The same state-tracking, FEN-assembly, and row-application upsampling can be
passed through the curriculum runner:

```powershell
.\sft\training\run-wsl.ps1 `
  -WslRepoPath /path/to/wsl/chess_sft_sdpo `
  -VenvPath /path/to/wsl/chess_sft_sdpo/.venv `
  chess-llm-run-curriculum --start-phase a --end-phase c `
  --task-upsample 1.5_state_tracking=4 `
  --task-upsample 1.9_fen_assembly=4 `
  --task-upsample 1.10_fen_row_application=4 `
  --wandb-project chess-sft `
  --wandb-group curriculum-fenassembly
```

The launcher syncs source files to WSL before each run, excluding `.git`,
local virtualenvs, generated artifacts, Polyglot binaries, pytest caches, and
`__pycache__`. Use `-NoSync` when you know the WSL copy is already current:

```powershell
.\sft\training\run-wsl.ps1 -NoSync -- pytest tests -q
```

## Evaluation Backends

Benchmark generation supports two backends:

- `transformers` - default, works with the same stack as training
- `vllm` - optional eval-only acceleration path

Select the backend from either CLI:

```bash
chess-llm-train --phase a --eval-only --inference-backend transformers
chess-llm-train --phase a --eval-only --inference-backend vllm
```

```bash
chess-llm-evaluate --model <checkpoint> --benchmark-dir <dir> --output preds.jsonl --inference-backend vllm
```

### `transformers`

Use this when:
- you want the simplest setup
- you are on Windows
- you want training and eval to share the same runtime stack

### `vllm`

Use this when:
- eval throughput matters
- you are running on a supported Linux or WSL GPU setup
- you want faster batched generation, especially for repeated benchmark runs

Notes:
- `vllm` is optional and is not part of the base `train` extra
- training still uses TRL / transformers
- only benchmark generation changes backends; scoring and gating stay the same

Example optional install:

```bash
pip install vllm
```

If `vllm` is requested but not installed, `chess-llm-evaluate` will raise a clear error.

## Benchmark Behavior

The benchmark uses:
- greedy generation for all gate metrics
- sampled generation only for extra `pass@k` candidates
- Stockfish ACPL only for Phase C planning by default
- a lightweight `*.analysis.json` sidecar for task-level failure examples and output-format bleed

Important details:
- Phase C `pass@8` uses the greedy answer plus 7 sampled candidates
- sampled transformers pass@k generation is batched with `num_return_sequences`
  instead of repeated full decode passes
- ACPL requires a valid Stockfish binary
- use `--full-acpl-report` to compute report-only ACPL on every split
- Phase C default eval covers the curriculum splits plus planning; auxiliary
  `chess960` and `mate` splits are reserved for `--full-benchmark`
- inspect `eval_predictions.analysis.json` after rehearsals to catch wrong answer formats, such as square/rank edit traces leaking into `board_to_fen`, `legal_moves`, or rule tasks
- `chess-llm-train` and `chess-llm-run-curriculum` use soft gates by default, so benchmark failures are logged but do not block the next phase
- use `--require-phase-gate` when Phase C should fail without ACPL or other required benchmark metrics
- benchmark regression checks compare against the best historical results from prior phases

Direct `chess-llm-evaluate` remains strict unless you pass `--soft-gate` or `--report-only`.

## W&B

Weights & Biases is enabled by default for real training runs.

Benchmark evaluation now logs there as well unless `--no-wandb` is set.

Keep `WANDB_API_KEY` set in the runtime environment or log in once with
`wandb login` before launching a real run. Disable logging only for CI,
offline smoke tests, or intentionally local-only runs:

```bash
chess-llm-train --phase a --no-wandb
```

If `WANDB_MODE=offline` is set, full training and direct eval fail fast so a
real run is not accidentally left local-only. Smoke runs allow offline W&B automatically;
for a non-smoke rehearsal, pass `--allow-wandb-offline`.

Set the project name with:

```bash
chess-llm-train --phase a --wandb-project chess-sft --wandb-group phase-a --run-name phase-a-full
chess-llm-run-curriculum --wandb-project chess-sft --wandb-group curriculum --run-prefix full-curriculum
```

## Common First-Run Flow

For a new machine or fresh environment:

```bash
chess-llm-train --phase a --dry-run
chess-llm-train --phase a --smoke-run --wandb-project chess-sft --run-name phase-a-smoke
chess-llm-train --phase a --wandb-project chess-sft --wandb-group phase-a --run-name phase-a-full
chess-llm-train --phase b --wandb-project chess-sft --wandb-group phase-b --run-name phase-b-full
chess-llm-train --phase c --wandb-project chess-sft --wandb-group phase-c --run-name phase-c-full
```

If you want faster eval on a Linux GPU box:

```bash
chess-llm-train --phase a --smoke-run --wandb-project chess-sft --run-name phase-a-vllm-smoke --inference-backend vllm
```

## Tests

Run the package training tests from the repo root:

```bash
python -m pytest tests -q
```

Current coverage includes:
- tier loading
- phase mixing
- checkpoint gating
- eval gate logic
- baseline merging
- smoke-run helper behavior
- eval subprocess command construction

## Known Environment Notes

- `trl` is required for actual training, but not for `--dry-run`
- `vllm` is optional and typically best on Linux / WSL GPU environments
- `python-chess` and Stockfish are required for ACPL
- if your environment has NumPy / pandas binary-compatibility issues, dataset loading may emit warnings before training starts
