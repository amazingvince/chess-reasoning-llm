# Chess LLM

This repository contains the package, tools, and runbooks for training and
evaluating chess-focused language models. The current loop is:

```text
generate SFT data -> train -> evaluate -> judge failures -> generate targeted data
```

The canonical implementation lives under `src/chess_llm/`. Console scripts are
declared in `pyproject.toml` and should be preferred over path-based Python
entry points.

## Current Shape

- `src/chess_llm/` contains the data, training, evaluation, artifact,
  Autodata, and shared chess utilities.
- `src/chess_llm_ui/` plus `ui/` contain the workbench backend and React UI.
- `docs/` contains current architecture, runbooks, schemas, research notes, and
  archived historical material.
- `sft/training/` is now only operational scaffolding: Docker files,
  PowerShell launchers, requirements files, and local environment templates.
- `polyglot_opening_books/` stores tracked Polyglot source archives. Extracted
  `.bin` books are local generated files and are ignored.
- `data/syzygy/` is the ignored local tablebase cache used by package defaults.

The old Python compatibility shims under `sft/make_data/` and `sft/training/`
have been removed. Use `chess_llm.*` imports and `chess-llm-*` commands.

## Docs

Start with [docs/index.md](docs/index.md).

Useful current docs:

- [Project map](docs/architecture/project_map.md)
- [CLI reference](docs/reference/cli.md)
- [Package layout](docs/reference/package_layout.md)
- [Phase A real-run runbook](docs/runbooks/phase_a_real_run.md)
- [Artifact schema](docs/schemas/artifact_v1.md)
- [Experiment log](docs/experiments/experiment_log.md)
- [UI workbench README](ui/README.md)

Historical plans, agent execution plans, and dated reviews live under
`docs/archive/`. They are useful context, but package settings and current
runbooks are the source of truth.

## Install

Install the root package in editable mode before using the CLIs:

```bash
python -m pip install -e ".[dev]"
```

Install optional extras for the workflow you are running:

```bash
python -m pip install -e ".[data]"
python -m pip install -e ".[train]"
python -m pip install -e ".[eval]"
python -m pip install -e ".[ui]"
```

For one WSL/Linux GPU environment that generates data, trains, and evaluates:

```bash
python -m pip install -e ".[data,train,eval]"
```

Stockfish remains an external binary. Pass `--stockfish-path` where supported
or set `STOCKFISH_PATH`.

## Main Commands

Data generation and validation:

```bash
chess-llm-preflight
chess-llm-extract-polyglot-books --polyglot-dir polyglot_opening_books
chess-llm-download-tablebases --pieces 3,4,5 --dry-run
chess-llm-make-data --volume 100 --all
chess-llm-make-data --tier 1 2
chess-llm-validate-outputs --output-dir chess_sft_data/output --blocklist chess_sft_data/eval_splits/blocklist.txt
chess-llm-upload-data --all --dry-run
```

Evaluation:

```bash
chess-llm-run-eval-split --volume 100
chess-llm-run-eval-harness --split-dir chess_sft_data/eval_splits
chess-llm-freeze-benchmark --split-dir chess_sft_data/eval_splits --output-dir chess_sft_data/benchmark
chess-llm-run-benchmark --benchmark-dir chess_sft_data/benchmark --predictions predictions.jsonl
chess-llm-batch-judge --benchmark-dir chess_sft_data/benchmark --predictions predictions.jsonl --output-dir artifacts --model-id Qwen/Qwen3-4B
chess-llm-sft-refresh --prompts artifacts/prompts.jsonl --rollouts artifacts/rollouts.jsonl --judgments artifacts/judgments.jsonl --output-dir chess_sft_data/autodata_refresh
```

Training:

```bash
chess-llm-train --phase a --dry-run
chess-llm-train --phase a --smoke-run --wandb-project chess-sft --run-name phase-a-smoke
chess-llm-evaluate --model chess_sft_checkpoints/phase_a/best --benchmark-dir chess_sft_data/benchmark --output predictions.jsonl --phase a
chess-llm-run-curriculum --start-phase a --end-phase c --wandb-project chess-sft --run-prefix full-curriculum
```

The `python -m chess_llm...` module form remains available for package modules,
but path-based legacy commands are retired.

## Current Curriculum Totals

The source of truth is `src/chess_llm/sft/settings.py`.

| Tier | Focus | Tasks | Target Rows |
| --- | --- | ---: | ---: |
| 1 | Perception, state, material mechanics | 19 | 1,560,000 |
| 2 | Rules and legal-move decomposition | 12 | 740,000 |
| 3 | Tactics | 5 | 260,000 |
| 4 | Evaluation | 3 | 150,000 |
| 5 | Openings | 3 | 30,000 |
| 6 | Endgames | 4 | 140,000 |
| 7 | Planning and verification | 6 | 270,000 |
| **Total** |  | **52** | **3,150,000** |

Phase A uses tiers 1-2: 31 tasks and 2,300,000 target rows before task
upsampling.

## Tests

Run root package tests:

```bash
python -m pytest tests -q
```

Run UI checks:

```bash
cd ui
npm test
npm run build
```
