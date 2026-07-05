# Decision-rule ledger design

## Feedback item

Before launching each experiment, record the keep/kill/escalate rule that will
govern the result. This prevents reinterpreting success after metrics arrive.

## Scope

- Add optional `--decision-rule TEXT` to direct evaluations.
- Add the same option to training so eval-only and post-training evals forward
  it into the evaluation subprocess.
- Store the rule in `EvaluationRunArtifact.metadata.decision_rule`.
- Let existing run-ledger and mirror finalization copy the finalized artifact.

## Non-goals

- Do not parse or enforce the rule in this slice.
- Do not change phase gates or metric thresholds.
- Do not require a decision rule for smoke tests or local debugging.
