# Project Map

## Direction

The project is now package-first. SFT remains the bootstrapping layer, but the
long-term center of gravity is a self-guided improvement loop:

```text
generate data -> train -> evaluate -> collect failures -> synthesize refresh data
```

Current code changes should land in `src/chess_llm/` or `src/chess_llm_ui/`.
Path-based compatibility modules under the old `sft/` trees have been retired.

## Package Boundaries

`src/chess_llm/` is the installable package namespace.

- `core`: stable chess primitives, FEN identity helpers, legal move helpers,
  rays, and opening-book utilities.
- `artifacts`: versioned persisted-record schemas and JSONL helpers.
- `formats`: prompt text, board rendering, answer parsing, and strict
  `<think>...</think><move>...</move>` protocol helpers.
- `sft`: SFT row contracts, settings, source loaders, generators,
  source-readiness checks, eval split construction, validation, output writing,
  upload helpers, and the main data-generation CLI.
- `evals`: eval split harnesses, frozen benchmark schema, benchmark scoring,
  prediction analysis, and batch judge artifact export.
- `training`: phase definitions, data loading/mixing, model loading, SFT
  trainer wiring, benchmark generation, curriculum execution, W&B helpers, and
  vLLM export.
- `autodata`: rollout artifacts, legality/Stockfish judgment, failure buckets,
  self-play scaffolding, and SFT refresh row generation.
- `external`: shared wrappers for external tools such as Stockfish and MultiPV.
- `preference`: preference-data interfaces.
- `sdpo`: feedback-distillation and on-policy training artifact namespace.
- `inference`: inference-facing package namespace.

`src/chess_llm_ui/` owns the FastAPI workbench backend. `ui/` owns the React
frontend.

## Operational Surfaces

`sft/training/` remains as launcher/runtime scaffolding only:

- `Dockerfile`, `Dockerfile.eval`, and `compose.yaml`
- `run-wsl.ps1`, `run-docker.ps1`, `run-eval-docker.ps1`
- `posthoc_eval_then_phase.ps1`
- `requirements.txt`, `requirements.eval.txt`, `.env.example`
- `build-causal-conv1d-wsl.sh`

These files launch package CLIs. They should not grow new Python application
logic.

Local generated or bulky assets use package-default roots:

- `polyglot_opening_books/` for tracked source archives and ignored extracted
  `.bin` books.
- `data/syzygy/` for ignored Syzygy tablebase files.
- `chess_sft_data/` for generated data and benchmark artifacts.
- `chess_sft_checkpoints/` for model checkpoints.

## Current Documentation

- `docs/index.md`: docs entry point.
- `docs/reference/package_layout.md`: package ownership details.
- `docs/reference/cli.md`: supported command surface.
- `docs/runbooks/phase_a_real_run.md`: current Phase A launch procedure.
- `docs/schemas/artifact_v1.md`: artifact schema reference.
- `docs/experiments/experiment_log.md`: dated run journal.

Historical plans, generated agent plans, and dated code reviews live under
`docs/archive/` and are not source-of-truth operational docs.

## Migration Rules

1. Use `chess_llm.*` imports for all Python code.
2. Use `chess-llm-*` console scripts for documented workflows.
3. Keep launcher scripts thin; they should invoke package CLIs.
4. Update `src/chess_llm/sft/settings.py` when task targets or default asset
   roots change, then update current docs that quote those values.
5. Keep root tests focused on package behavior. Do not reintroduce tests whose
   only purpose is proving retired shim imports.
