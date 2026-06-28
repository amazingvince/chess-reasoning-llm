# Self-Guided Chess Improvement Loop

## Loop

The target system is a recurring improvement loop:

```text
source data
  -> SFT/bootstrap datasets
  -> training checkpoint
  -> static and dynamic evaluation
  -> model rollouts on selected positions
  -> judge with python-chess, Stockfish, Syzygy, puzzle oracles, and verifiers
  -> failure buckets and teacher feedback
  -> targeted SFT refresh, preference pairs, or feedback-distillation examples
  -> next checkpoint
```

## Stages

1. **Bootstrap chess competence**
   - Use the existing SFT curriculum to teach FEN, legal moves, tactics,
     evaluation, openings, endgames, and move selection.
   - Keep a small amount of rules replay in later stages to prevent format and
     legality drift.

2. **Evaluate before generating more data**
   - Run frozen static benchmarks and, later, arena games.
   - Record raw model outputs, parsed moves, legal status, regret, and failure
     labels as artifacts.
   - The current concrete layer supports prompt -> rollout -> legality
     judgment, with optional Stockfish regret scoring for legal move outputs.

3. **Target the model's actual failures**
   - Select positions where the current checkpoint fails and the teacher is
     confident.
   - Prefer learning signal over generic difficulty.

4. **Build training artifacts from judgments**
   - SFT refresh examples teach corrected behavior.
   - Preference pairs compare the model's mistake with a better response.
   - Feedback-distillation examples condition a teacher or self-teacher on
     verifier feedback.

5. **Train and compare**
   - Train a short refresh or alignment run.
   - Compare against the previous checkpoint on frozen benchmarks, held-out
     failure buckets, and later arena games.

## Artifact Flow

- `PromptArtifact`: position and task sent to a model.
- `RolloutArtifact`: raw model output plus parsed answer.
- `JudgmentArtifact`: legality, regret, failure bucket, teacher move, feedback.
- `PreferencePairArtifact`: chosen/rejected responses for DPO-style training.
- `FeedbackDistillationArtifact`: student output, feedback, and corrected target
  for SDPO or related on-policy distillation.

These records let the project audit where each training example came from and
why it should improve the model.

## Current Bootstrap Layer

The first Autodata substrate is intentionally narrow:

1. Build a `RolloutArtifact` from `PromptArtifact`, model ID, and raw text.
2. Parse the raw text with the flexible answer parser.
3. Judge missing FEN, parse failure, illegal move, or legal unscored move.
4. Persist prompt, rollout, and judgment rows as JSONL.

This is enough to audit model legality, parse behavior, and first-pass
Stockfish regret before adding direct Syzygy WDL/DTZ probing, preference-pair
construction, or SDPO feedback generation.
