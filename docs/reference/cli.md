# CLI Reference

Install the package from the repo root before running these commands:

```bash
python -m pip install -e ".[data,train,eval]"
```

## Data

| Command | Module | Purpose |
| --- | --- | --- |
| `chess-llm-make-data` | `chess_llm.sft.pipeline` | Generate SFT data, eval splits, and benchmark files. |
| `chess-llm-preflight` | `chess_llm.sft.preflight` | Check local data, engines, and generated assets. |
| `chess-llm-validate-outputs` | `chess_llm.sft.validate_outputs` | Validate generated tier JSONL and decontamination. |
| `chess-llm-upload-data` | `chess_llm.sft.hub_upload` | Stage or upload generated datasets to Hugging Face Hub. |
| `chess-llm-extract-polyglot-books` | `chess_llm.sft.extract_polyglot_books` | Extract local Polyglot archives into ignored `.bin` books. |
| `chess-llm-download-tablebases` | `chess_llm.sft.download_tablebases` | Download Syzygy tablebase files into `data/syzygy/`. |
| `chess-llm-build-candidate-ratings` | `chess_llm.sft.candidate_ratings` | Build candidate-rating annotation inputs. |

## Evaluation And Artifacts

| Command | Module | Purpose |
| --- | --- | --- |
| `chess-llm-run-eval-split` | `chess_llm.sft.run_eval_split` | Refresh held-out eval splits without full generation. |
| `chess-llm-run-eval-harness` | `chess_llm.evals.run_eval_harness` | Validate raw eval-split oracle consistency. |
| `chess-llm-freeze-benchmark` | `chess_llm.evals.freeze_benchmark` | Convert eval splits into frozen benchmark examples. |
| `chess-llm-run-benchmark` | `chess_llm.evals.run_benchmark` | Score prediction JSONL against a frozen benchmark. |
| `chess-llm-batch-judge` | `chess_llm.evals.batch_judge` | Export prompt, rollout, judgment, and manifest artifacts. |
| `chess-llm-sft-refresh` | `chess_llm.autodata.sft_refresh` | Convert judged failures into targeted SFT refresh rows. |
| `chess-llm-r0-report` | `chess_llm.training.r0_gate` | Report R0 gate diagnostics. |

`chess-llm-evaluate` and `chess-llm-train` both accept
`--run-ledger PATH` and `--artifact-mirror-dir DIR`. The ledger receives one
finalized evaluation-run JSON row per invocation; the mirror stores existing
prediction, results, analysis, eval-run, and MultiPV cache files under
`DIR/<eval_run_id>/`.

## Training

| Command | Module | Purpose |
| --- | --- | --- |
| `chess-llm-train` | `chess_llm.training.train` | Run phase-aware SFT training or benchmark eval from the training CLI. |
| `chess-llm-evaluate` | `chess_llm.training.evaluate` | Run direct benchmark generation/scoring for a model checkpoint. |
| `chess-llm-run-curriculum` | `chess_llm.training.run_curriculum` | Chain phase training and post-train evaluation. |
| `chess-llm-export-vllm` | `chess_llm.training.vllm_export` | Export compatible checkpoint layout for vLLM when needed. |

## Notes

- Path-based legacy commands under `sft/make_data/scripts` and
  `sft/training/*.py` are retired.
- `sft/training/run-wsl.ps1`, `run-docker.ps1`, and `run-eval-docker.ps1`
  remain supported launch helpers for package CLIs.
- Many modules can still be invoked with `python -m chess_llm...` when direct
  module execution is useful for debugging.
