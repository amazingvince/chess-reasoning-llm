# Project Map

## Direction

The project is moving from an SFT-only repository toward a chess model
improvement system. SFT remains the bootstrapping layer, but the long-term
center of gravity is the self-guided loop: collect model rollouts, judge
failures, synthesize targeted data, and train the next checkpoint.

## Package Boundaries

`src/chess_llm/` is the package namespace for shared and future-facing code.

- `core`: stable chess/project primitives shared across subsystems.
- `artifacts`: versioned schemas and JSONL utilities for persisted records.
- `formats`: model input/output parsing and normalization.
- `sft`: package-owned SFT row contracts, prompt templates, task generators,
  validation, source helpers, output auditing, and the main data-generation
  pipeline. Some ops scripts still live under `sft/make_data` behind
  compatibility imports while migration continues.
- `evals`: static benchmark and future arena evaluation interfaces.
- `autodata`: challenger, solver, judge, and recipe-optimization interfaces.
- `external`: shared wrappers for external engines and tools such as Stockfish.
- `preference`: chosen/rejected pair construction and preference datasets.
- `sdpo`: feedback-distillation and on-policy training artifacts.
- `inference`: fast move, reasoning move, and tutor-facing interfaces.

## Legacy Compatibility

The active SFT and training implementations now live under `src/chess_llm`.
The old trees remain as compatibility surfaces while the package interfaces
settle:

- `sft/make_data/` keeps legacy imports, docs, tests, and ops scripts that
  forward into package-owned SFT data, eval split, benchmark, validation, and
  source helper modules.
- `sft/training/` keeps legacy imports, docs, tests, and wrappers that forward
  into package-owned phase data loading, training, evaluation, and curriculum
  modules.

Future migrations should remove compatibility code only after its package
entry point, tests, and docs are stable. Avoid moving or deleting old files just
to make the tree look cleaner.

## Migration Path

1. Use `chess_llm.artifacts` for any new persisted JSONL records.
2. Use `chess_llm.formats.answers.parse_answer` before judging model outputs.
3. Add Autodata and preference builders against the artifact schemas.
4. Introduce adapters from the existing benchmark/eval code to artifact
   records.
5. Keep source-readiness, package CLI, and wheel-install checks green before
   deleting legacy wrappers.
