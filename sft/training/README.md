# training - Chess SFT Training Harness

Three-phase supervised fine-tuning harness for the chess SFT curriculum in `sft/make_data/output/`.

This package handles:
- phase-aware dataset mixing across tiers 1-7
- chained A -> B -> C training
- benchmark evaluation and phase gating
- bounded smoke runs for end-to-end verification
- optional `transformers` or `vllm` inference for eval

Default base model: `Qwen/Qwen3-0.6B`

Override it with:

```bash
set CHESS_SFT_BASE_MODEL=Qwen/Qwen3-1.7B
```

## Quick Start

Run these commands from `sft/training/`.

```bash
# Install base dependencies
pip install -r requirements.txt

# Inspect phase config and expected data counts
python train.py --phase a --dry-run

# Short end-to-end smoke test
python train.py --phase a --smoke-run --no-wandb

# Full Phase A training
python train.py --phase a --no-wandb

# Full Phase B training (uses Phase A best/ checkpoint if present)
python train.py --phase b --no-wandb

# Full Phase C training
python train.py --phase c --no-wandb

# Full end-to-end curriculum: train -> post-eval telemetry across A/B/C
python run_curriculum.py --wandb-project chess-sft
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
|-- train.py              # Main training CLI
|-- evaluate.py           # Frozen benchmark evaluation CLI
|-- run_curriculum.py     # End-to-end A -> B -> C orchestration
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
data root:       E:/chess_sft_data/output
benchmark root:  E:/chess_sft_data/benchmark
checkpoints:     E:/chess_sft_checkpoints
```

Phase outputs are written under:

```text
<output-root>/
|-- phase_a/
|   |-- best/
|   |-- eval_predictions.jsonl
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
python train.py --phase a --dry-run
python train.py --phase c --dry-run --inference-backend vllm
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
python train.py --phase a --smoke-run --no-wandb
```

You can also override the bounds directly:

```bash
python train.py --phase a \
  --max-train-examples 256 \
  --max-eval-examples 64 \
  --max-benchmark-examples-per-split 50 \
  --max-steps 20 \
  --no-wandb
```

### Trainer Eval Defaults

Full runs use a bounded trainer-eval subset for loss tracking and best-checkpoint selection:
- trainer eval examples: `2048` by default
- trainer eval/save cadence: every `5000` optimizer steps
- benchmark eval remains telemetry by default

Useful controls:
- `--max-eval-examples 0` to use the full mixed eval set during training
- `--trainer-eval-steps 2000 --trainer-save-steps 2000` for more frequent best-checkpoint checks
- `--skip-trainer-eval` to train without in-loop eval and save the final checkpoint to `best/`

### Eval Only

Run benchmark evaluation without training:

```bash
python train.py --phase a --eval-only
python train.py --phase c --eval-only --inference-backend vllm
```

Behavior:
- if `phase_<name>/best/` exists, it evaluates that checkpoint
- otherwise it falls back to the phase starting model

Evaluation runs now support W&B logging too. Use `--no-wandb` to disable it.

### Full Curriculum Runner

Run train and post-train benchmark telemetry for each phase:

```bash
python run_curriculum.py --wandb-project chess-sft
```

Useful options:
- `--start-phase b --end-phase c` to resume from a later stage
- `--pre-eval` to add a lightweight report-only eval before each phase
- `--require-phase-gate` to make benchmark failures block progression
- `--smoke-run` to do a bounded end-to-end rehearsal
- `--run-prefix march19` to stamp consistent run names

The runner performs:
- training for that phase
- post-train benchmark eval via `train.py`
- automatic stop on training/runtime failures; benchmark metric failures stop only with `--require-phase-gate`

### Direct Benchmark Eval

You can also call `evaluate.py` directly:

```bash
python evaluate.py \
  --model E:/chess_sft_checkpoints/phase_a/best \
  --benchmark-dir E:/chess_sft_data/benchmark \
  --output phase_a_predictions.jsonl \
  --phase a
```

Phase C with pass@8:

```bash
python evaluate.py \
  --model E:/chess_sft_checkpoints/phase_c/best \
  --benchmark-dir E:/chess_sft_data/benchmark \
  --output phase_c_predictions.jsonl \
  --phase c \
  --pass-k 8
```

## Docker Training

You can run the harness inside a Linux GPU container and pin it to the 5090.

Files:
- [Dockerfile](/c:/Users/amazi/code/chess_sft_sdpo/sft/training/Dockerfile)
- [compose.yaml](/c:/Users/amazi/code/chess_sft_sdpo/sft/training/compose.yaml)
- [.env.example](/c:/Users/amazi/code/chess_sft_sdpo/sft/training/.env.example)
- [run-docker.ps1](/c:/Users/amazi/code/chess_sft_sdpo/sft/training/run-docker.ps1)

The compose setup defaults to:
- `CUDA_VISIBLE_DEVICES=1`
- data root mounted from `E:/chess_sft_data`
- checkpoints mounted from `E:/chess_sft_checkpoints`
- HuggingFace cache mounted from `E:/hf_cache`
- `BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`
- `transformers>=5.3.0,<5.13.0`
- `INSTALL_VLLM=0`
- `INSTALL_FLASH_ATTN=1`

On this machine, Linux CUDA enumerates:
- CUDA device `0` -> RTX 4090
- CUDA device `1` -> RTX 5090

So the Docker defaults target the 5090 by setting `CUDA_DEVICE_ID=1`.

### First-Time Setup

1. Start Docker Desktop.
2. Make sure Docker Desktop is using Linux containers with the WSL2 backend.
3. Copy `.env.example` to `.env` in `sft/training/` and adjust paths if needed.

```bash
cd sft/training
Copy-Item .env.example .env
```

### Build the Image

From `sft/training/`:

```bash
docker compose build
```

If you want `vllm` installed in the image too:

```bash
$env:INSTALL_VLLM="1"
docker compose build
```

Flash Attention 2 is installed from the matching published wheel by default on this machine:

```bash
$env:INSTALL_FLASH_ATTN="1"
docker compose build
```

The default training image uses the exact wheel-backed combo that was verified locally:
- base image: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`
- Python: `3.11`
- Torch: `2.8.0+cu128`
- FlashAttention wheel: `flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl`

If FlashAttention is missing or the extension cannot import, model loading falls back to explicit PyTorch SDPA.

### Recommended Way to Run on This Machine

Use the PowerShell helper:

```bash
.\run-docker.ps1 python train.py --phase a --smoke-run --no-wandb
```

Full end-to-end curriculum in Docker:

```bash
.\run-docker.ps1 python run_curriculum.py --wandb-project chess-sft
```

That helper:
- mounts the repo and data directories
- requests GPU access from Docker
- sets `CUDA_VISIBLE_DEVICES=1`
- defaults to the 5090 on this machine

### Verify GPU Access

Open a shell in the container:

```bash
.\run-docker.ps1
```

Then inside the container:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Run a Smoke Test in Docker

```bash
.\run-docker.ps1 python train.py --phase a --smoke-run --no-wandb
```

### Run Full Training on the 5090

```bash
.\run-docker.ps1 python train.py --phase a --no-wandb
.\run-docker.ps1 python train.py --phase b --no-wandb
.\run-docker.ps1 python train.py --phase c --no-wandb
```

### Eval in Docker

Transformers backend:

```bash
.\run-docker.ps1 python evaluate.py \
  --model /data/chess_sft_checkpoints/phase_a/best \
  --benchmark-dir /data/chess_sft_data/benchmark \
  --output /data/chess_sft_checkpoints/phase_a/eval_predictions.jsonl \
  --phase a \
  --inference-backend transformers
```

vLLM backend:

```bash
.\run-docker.ps1 python evaluate.py \
  --model /data/chess_sft_checkpoints/phase_a/best \
  --benchmark-dir /data/chess_sft_data/benchmark \
  --output /data/chess_sft_checkpoints/phase_a/eval_predictions.jsonl \
  --phase a \
  --inference-backend vllm
```

### Notes

- The container installs Linux `stockfish` and sets `STOCKFISH_PATH=/usr/games/stockfish`.
- Training still uses TRL / transformers.
- `vllm` is optional and off by default.
- Change `CUDA_DEVICE_ID` in `.env` if you want to target a different CUDA device later.
- Docker bind mounts from Windows paths work, but for best filesystem performance you may eventually want the repo and caches inside the WSL/Linux filesystem.

## WSL Training

The PowerShell launcher [run-wsl.ps1](/c:/Users/amazi/code/chess_sft_sdpo/sft/training/run-wsl.ps1) runs training from a Linux-side source copy while keeping large artifacts on `E:`.

Current local layout:

```text
WSL distro:       Ubuntu-20.04
WSL repo:         /home/vince/code/chess_sft_sdpo
Conda env:        chess-sft-wsl
Data root:        /mnt/e/chess_sft_data
Checkpoints:      /mnt/e/chess_sft_checkpoints
HF cache:         /mnt/e/hf_cache
W&B cache:        /mnt/e/wandb_cache
Stockfish:        /mnt/e/stockfish/stockfish
PyTorch GPU id:   CUDA_VISIBLE_DEVICES=1 -> RTX 5090
```

The current Ubuntu-20.04 WSL image uses SDPA by default. The published FlashAttention 2.8.3 wheel imports against a newer glibc than Ubuntu-20.04 provides, so use `--attn-implementation auto` and let the loader fall back, build FlashAttention from source, or move this training env to Ubuntu 22.04/24.04 if FA2 is required.

Run commands from the Windows repo root:

```powershell
.\sft\training\run-wsl.ps1 python train.py --phase a --dry-run --no-wandb
.\sft\training\run-wsl.ps1 python train.py --phase a --smoke-run --no-wandb
.\sft\training\run-wsl.ps1 python train.py --phase a --no-wandb
```

The launcher syncs source files to WSL before each run, excluding generated tablebases, Polyglot binaries, pytest caches, and `__pycache__`. Use `-NoSync` when you know the WSL copy is already current:

```powershell
.\sft\training\run-wsl.ps1 -NoSync pytest tests -q
```

## Evaluation Backends

Benchmark generation supports two backends:

- `transformers` - default, works with the same stack as training
- `vllm` - optional eval-only acceleration path

Select the backend from either CLI:

```bash
python train.py --phase a --eval-only --inference-backend transformers
python train.py --phase a --eval-only --inference-backend vllm
```

```bash
python evaluate.py --model <checkpoint> --benchmark-dir <dir> --output preds.jsonl --inference-backend vllm
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
- `vllm` is optional and is not part of the base `requirements.txt`
- training still uses TRL / transformers
- only benchmark generation changes backends; scoring and gating stay the same

Example optional install:

```bash
pip install vllm
```

If `vllm` is requested but not installed, `evaluate.py` will raise a clear error.

## Benchmark Behavior

The benchmark uses:
- greedy generation for all gate metrics
- sampled generation only for extra `pass@k` candidates
- Stockfish ACPL only for Phase C planning by default

Important details:
- Phase C `pass@8` uses the greedy answer plus 7 sampled candidates
- ACPL requires a valid Stockfish binary
- use `--full-acpl-report` to compute report-only ACPL on every split
- `train.py` and `run_curriculum.py` use soft gates by default, so benchmark failures are logged but do not block the next phase
- use `--require-phase-gate` when Phase C should fail without ACPL or other required benchmark metrics
- benchmark regression checks compare against the best historical results from prior phases

Direct `evaluate.py` remains strict unless you pass `--soft-gate` or `--report-only`.

## W&B

Weights & Biases is enabled by default for real training runs.

Benchmark evaluation now logs there as well unless `--no-wandb` is set.

Disable it with:

```bash
python train.py --phase a --no-wandb
```

Set the project name with:

```bash
python train.py --phase a --wandb-project chess-sft
python run_curriculum.py --wandb-project chess-sft --run-prefix full-curriculum
```

## Common First-Run Flow

For a new machine or fresh environment:

```bash
python train.py --phase a --dry-run
python train.py --phase a --smoke-run --no-wandb
python train.py --phase a --no-wandb
python train.py --phase b --no-wandb
python train.py --phase c --no-wandb
```

If you want faster eval on a Linux GPU box:

```bash
python train.py --phase a --smoke-run --no-wandb --inference-backend vllm
```

## Tests

Run the training harness test suite from `sft/training/`:

```bash
pytest tests -q
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
