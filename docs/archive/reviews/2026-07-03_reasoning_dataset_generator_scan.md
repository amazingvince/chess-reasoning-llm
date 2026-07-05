# reasoning_dataset_generator Scan Notes

Reviewed source: https://github.com/amazingvince/reasoning_dataset_generator.git at commit `e58be9d153fec1c410cdba552fb0b5bd05e1ef9f`.

## Keep As Ideas

- MultiPV-backed candidate tables and cached top-N move evaluations.
- Shallow-vs-deep trap/resource detection for candidate scan examples.
- Win Probability Delta scoring as a future reward/diagnostic shape.
- Per-prompt diversity telemetry: unique parsed moves, mode share, parse rate, reward standard deviation.
- Streaming JSONL sharding/upload ergonomics for long-running data generation.

## Do Not Import

- The free-form trace generator: it conflicts with the fixed-grammar, machine-verifiable direction in `plan/06_reasoning_expansion.md`.
- The standalone SFT/GRPO/UI pipeline: this repo already owns those surfaces.
- Direct code: the nested checkout had no top-level license file.

## Decision

Delete the nested checkout and keep this note as the only retained artifact.
