# Artifact Schema v1

All persisted loop records should be JSONL rows using
`schema_version: "artifact.v1"`. The schema is implemented in
`chess_llm.artifacts.schemas`.

## Common Fields

- `schema_version`: currently `artifact.v1`.
- `artifact_type`: record kind, such as `prompt`, `rollout`, or `judgment`.
- `metadata`: free-form object for run IDs, source names, sampling settings, or
  experiment notes.

## ChatMessage

Nested chat message used by prompt artifacts.

```json
{"role":"user","content":"FEN: 8/8/8/8/8/8/4K3/4k3 w - - 0 1"}
```

## PromptArtifact

Prompt prepared for evaluation, rollout, or data generation.

```json
{
  "schema_version": "artifact.v1",
  "artifact_type": "prompt",
  "prompt_id": "prompt-001",
  "messages": [
    {"role": "system", "content": "You are a chess move-selection model."},
    {"role": "user", "content": "FEN: ..."}
  ],
  "fen": "8/8/8/8/8/8/4K3/4k3 w - - 0 1",
  "task_type": "best_move",
  "metadata": {"split": "planning"}
}
```

## ParsedAnswer

Normalized answer parsed from raw model text. The raw text is preserved
elsewhere; this record stores the extracted move and parse status.

```json
{
  "schema_version": "artifact.v1",
  "artifact_type": "parsed_answer",
  "raw_text": "<think>Take the center.</think>\n<move>e2e4</move>",
  "move_uci": "e2e4",
  "format_type": "move_tag",
  "parse_error": null,
  "metadata": {}
}
```

Supported parser formats are `move_tag`, `json`, `bare_uci`, `prose_uci`, and
`none`. Ambiguous prose with multiple different UCI moves is recorded as
`format_type: "none"`.

## RolloutArtifact

One model response to one prompt.

```json
{
  "schema_version": "artifact.v1",
  "artifact_type": "rollout",
  "rollout_id": "rollout-001",
  "prompt_id": "prompt-001",
  "model_id": "Qwen/Qwen3-4B",
  "raw_output": "<think>Take the center.</think>\n<move>e2e4</move>",
  "parsed_answer": {
    "schema_version": "artifact.v1",
    "artifact_type": "parsed_answer",
    "raw_text": "<think>Take the center.</think>\n<move>e2e4</move>",
    "move_uci": "e2e4",
    "format_type": "move_tag",
    "parse_error": null,
    "metadata": {}
  },
  "metadata": {"temperature": 0.2}
}
```

## JudgmentArtifact

Verifier or teacher assessment of a rollout.

```json
{
  "schema_version": "artifact.v1",
  "artifact_type": "judgment",
  "judgment_id": "judgment-001",
  "rollout_id": "rollout-001",
  "legal": true,
  "regret_cp": 34.5,
  "failure_bucket": null,
  "teacher_move_uci": "e2e4",
  "feedback": "The move is legal and close to best.",
  "metadata": {"judge": "stockfish"}
}
```

## PreferencePairArtifact

Chosen/rejected response for preference training.

```json
{
  "schema_version": "artifact.v1",
  "artifact_type": "preference_pair",
  "pair_id": "pair-001",
  "prompt_id": "prompt-001",
  "chosen_output": "<move>e2e4</move>",
  "rejected_output": "<move>e2e3</move>",
  "chosen_move_uci": "e2e4",
  "rejected_move_uci": "e2e3",
  "reason": "Chosen move has lower engine regret.",
  "metadata": {}
}
```

## FeedbackDistillationArtifact

Student output, feedback, and corrected target for SDPO-style or related
feedback-conditioned distillation.

```json
{
  "schema_version": "artifact.v1",
  "artifact_type": "feedback_distillation",
  "example_id": "feedback-001",
  "prompt_id": "prompt-001",
  "student_output": "<move>e2e3</move>",
  "feedback": "Best move is e2e4 because it controls the center.",
  "teacher_output": "<think>Take the center.</think>\n<move>e2e4</move>",
  "target_output": "<think>Take the center.</think>\n<move>e2e4</move>",
  "metadata": {}
}
```

## Training Consumers

- SFT refresh consumes prompts plus corrected target outputs.
- Preference training consumes `PreferencePairArtifact` rows.
- SDPO and related feedback-distillation experiments consume
  `FeedbackDistillationArtifact` rows.

## Minimal Bootstrap Flow

The current Autodata bootstrap can persist these three rows in separate JSONL
files or join them by ID.

Prompt:

```json
{
  "schema_version": "artifact.v1",
  "artifact_type": "prompt",
  "prompt_id": "prompt-001",
  "messages": [
    {"role": "user", "content": "Choose a legal move from the FEN."}
  ],
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "task_type": "best_move",
  "metadata": {}
}
```

Rollout:

```json
{
  "schema_version": "artifact.v1",
  "artifact_type": "rollout",
  "rollout_id": "rollout-001",
  "prompt_id": "prompt-001",
  "model_id": "Qwen/Qwen3-4B",
  "raw_output": "<move>e2e4</move>",
  "parsed_answer": {
    "schema_version": "artifact.v1",
    "artifact_type": "parsed_answer",
    "raw_text": "<move>e2e4</move>",
    "move_uci": "e2e4",
    "format_type": "move_tag",
    "parse_error": null,
    "metadata": {}
  },
  "metadata": {}
}
```

Judgment:

```json
{
  "schema_version": "artifact.v1",
  "artifact_type": "judgment",
  "judgment_id": "judgment-001",
  "rollout_id": "rollout-001",
  "legal": true,
  "regret_cp": null,
  "failure_bucket": "legal_unscored",
  "teacher_move_uci": null,
  "feedback": "Parsed move e2e4 is legal; engine regret is not scored yet.",
  "metadata": {}
}
```
