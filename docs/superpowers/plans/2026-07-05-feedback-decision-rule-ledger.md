# Decision-rule ledger feedback plan

## Task 1: Red tests

Files:

- `tests/test_training_eval_reliability.py`
- `tests/test_training_entrypoints.py`

Steps:

- [x] Add a failing test for direct eval-run metadata containing `decision_rule`.
- [x] Add a failing test for training eval-command forwarding.

## Task 2: Implementation

Files:

- `src/chess_llm/training/evaluate.py`
- `src/chess_llm/training/train.py`

Steps:

- [x] Add `EvaluationConfig.decision_rule`.
- [x] Add `chess-llm-evaluate --decision-rule`.
- [x] Store the rule in evaluation-run metadata.
- [x] Add `chess-llm-train --decision-rule`.
- [x] Forward the rule to eval-only and post-training eval commands.

## Task 3: Verification and commit

Files:

- All files above plus docs.

Steps:

- [x] Run focused decision-rule tests.
- [x] Run broader eval/training entrypoint tests.
- [x] Run the full test suite.
- [x] Inspect staged diff.
- [x] Commit the decision-rule slice.
