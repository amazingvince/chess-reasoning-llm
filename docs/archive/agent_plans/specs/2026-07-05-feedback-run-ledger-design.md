# Feedback Run Ledger And Artifact Mirror Design

## Goal

Make benchmark evaluation artifacts durable and indexable immediately after each run.

## Scope

This slice adds two optional paths to `chess-llm-evaluate`:

- `--run-ledger PATH`: append one `EvaluationRunArtifact` JSON row to a JSONL ledger.
- `--artifact-mirror-dir DIR`: copy prediction sidecars into `DIR/<eval_run_id>/`.

`chess-llm-train` forwards the same paths to eval-only and post-training evaluation subprocesses. The implementation covers both successful evals and infrastructure failures that still write an eval-run sidecar.

This slice does not implement remote rsync/SCP, cloud object storage, run decision rules, or generated comparison tables. It provides the local durable mirror and ledger those follow-on tools can consume.

## Artifact Policy

The mirror copies every existing sidecar from the eval output family:

- predictions JSONL,
- results JSON,
- prediction analysis JSON,
- eval-run metadata JSON,
- MultiPV SQLite cache when present.

The eval-run metadata is finalized before copying so the mirrored `*.eval_run.json` contains:

- `metadata.run_ledger_path`,
- `metadata.artifact_mirror_root`,
- `metadata.artifact_mirror_dir`,
- `metadata.mirrored_artifacts`.

The ledger appends the finalized eval-run JSON object. It intentionally does not deduplicate rows; reruns or repeated finalization calls remain visible as separate attempts.

## Tests

Add focused tests for:

- ledger append and mirror copy behavior from a standalone eval-run sidecar,
- append-only ledger behavior without a mirror directory,
- `run_evaluation` writing ledger and mirror artifacts,
- `chess-llm-train` forwarding ledger and mirror flags into the eval command.
