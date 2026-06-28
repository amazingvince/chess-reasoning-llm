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
- `sft`: adapters around the existing SFT bootstrap pipeline.
- `evals`: static benchmark and future arena evaluation interfaces.
- `autodata`: challenger, solver, judge, and recipe-optimization interfaces.
- `external`: shared wrappers for external engines and tools such as Stockfish.
- `preference`: chosen/rejected pair construction and preference datasets.
- `sdpo`: feedback-distillation and on-policy training artifacts.
- `inference`: fast move, reasoning move, and tutor-facing interfaces.

## Existing Code

The current working code remains in place for this pass:

- `sft/make_data/` builds SFT data, eval splits, benchmarks, and validators.
- `sft/training/` trains and evaluates staged SFT checkpoints.

Future migrations should move shared logic into `chess_llm` only when it has a
clear cross-subsystem consumer. Avoid moving working SFT modules just to make
the tree look cleaner.

## Migration Path

1. Use `chess_llm.artifacts` for any new persisted JSONL records.
2. Use `chess_llm.formats.answers.parse_answer` before judging model outputs.
3. Add Autodata and preference builders against the artifact schemas.
4. Introduce adapters from the existing benchmark/eval code to artifact
   records.
5. Move legacy modules only after the package interfaces have stabilized.
