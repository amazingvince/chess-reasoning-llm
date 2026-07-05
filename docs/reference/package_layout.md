# Package Layout

The installable package is rooted at `src/chess_llm/`.

| Package | Responsibility |
| --- | --- |
| `chess_llm.core` | Chess primitives, FEN identity, legality helpers, opening-book utilities. |
| `chess_llm.formats` | Prompt text, board rendering, answer parsing, and move-tag protocols. |
| `chess_llm.artifacts` | Versioned JSONL artifact schemas and readers/writers. |
| `chess_llm.sft` | SFT row contracts, generators, source loaders, source readiness, validation, output writing, upload, and data-generation orchestration. |
| `chess_llm.evals` | Eval split checks, benchmark freezing/scoring, prediction analysis, and batch judgment export. |
| `chess_llm.training` | Phase definitions, data loading/mixing, model loading, SFT trainer wiring, direct eval, curriculum runner, W&B helpers, and vLLM export. |
| `chess_llm.autodata` | Rollout construction, judging, self-play scaffolding, failure buckets, and SFT refresh row generation. |
| `chess_llm.external` | External engine wrappers such as Stockfish and MultiPV helpers. |
| `chess_llm.preference` | Preference-training data interfaces. |
| `chess_llm.sdpo` | SDPO and feedback-distillation artifact namespace. |
| `chess_llm.inference` | Inference-facing package namespace. |

The workbench backend package is separate:

| Package | Responsibility |
| --- | --- |
| `chess_llm_ui` | FastAPI backend, local LLM client, artifact review helpers, and tool endpoints for the React UI. |

## Retired Shims

The Python compatibility shims under `sft/make_data/` and `sft/training/` have
been deleted. New code should import from `chess_llm.*` and use console
scripts from `pyproject.toml`.

`sft/training/` remains for operational files only: Dockerfiles, Compose,
PowerShell launchers, requirements, and environment templates.

Local generated assets use these package defaults:

- Polyglot archives and extracted books: `polyglot_opening_books/`
- Syzygy tablebases: `data/syzygy/`
- Generated SFT data: `chess_sft_data/`
- Checkpoints: `chess_sft_checkpoints/`
