# Live Play Reliability Loop Design

## Summary

Improve the Chess LLM Workbench live-play flow so model failures are explicit,
recoverable, and useful for debugging. V1 focuses on illegal or unparsed LLM
responses during live games. The app preserves the failed model output as
research evidence, keeps the board on the same position, and lets the user
choose how to recover.

## Goals

- Make the active LLM runtime state clear: mode, model, and last request result.
- Keep illegal or unparsed LLM responses visible instead of silently hiding them.
- Prevent illegal LLM responses from advancing the board.
- Provide explicit recovery actions:
  - Retry LLM.
  - Retry LLM with legal moves emphasized.
  - Use teacher move when available.
  - Use deterministic legal move.
  - Leave the game paused.
- Persist failed attempts and successful recovery moves as artifacts with enough
  metadata to review them later.

## Non-Goals

- No automatic fallback in V1. The user must choose a recovery action.
- No rollout metrics dashboard in this slice.
- No full two-board move inspector in this slice.
- No Stockfish installation or download workflow.
- No rewriting source artifact directories loaded in review mode.

## Current Behavior

`POST /api/games/{game_id}/llm-move` asks the configured LLM client for a move,
judges the parsed output, persists prompt/rollout/judgment rows, and returns a
`GameState`. When the parsed move is legal, the board advances. When the parsed
move is illegal, the backend records the rollout and judgment but does not push
the move onto the board. The UI shows the failed judgment, but recovery is manual
and unclear.

## Proposed User Flow

1. User clicks `Ask LLM`.
2. Backend builds the prompt, calls the LLM, parses output, judges the rollout,
   and writes artifacts.
3. If the judgment is legal, the move is applied as today.
4. If the judgment is illegal or the parsed move is missing:
   - The board remains at the same FEN.
   - The failed rollout and judgment remain visible in the inspector.
   - The game enters a paused recovery state.
   - The UI shows a recovery panel with explicit next actions.
5. User chooses one recovery action:
   - `Retry LLM`: call the normal LLM prompt again.
   - `Retry with legal moves`: call the LLM with a stronger legal-move list in
     the prompt metadata/content.
   - `Use teacher move`: apply `teacher_move_uci` from the latest judgment when
     present and legal.
   - `Use deterministic legal move`: use the local deterministic legal stub for
     the current position.
   - `Leave paused`: take no backend action.
6. Any recovery attempt that returns or applies a legal move advances the board
   and clears the paused recovery state.
7. Any failed recovery attempt keeps the board paused and shows the latest failed
   attempt.

## Backend API

Add:

```http
POST /api/games/{game_id}/recovery-move
```

Request body:

```json
{
  "action": "retry" | "retry_with_legal_moves" | "teacher" | "legal_stub"
}
```

Response:

- `200 OK` with updated `GameState` when the action is accepted.
- `400 Bad Request` when there is no active failed LLM attempt, the selected
  recovery action is unavailable, or the recovery move is illegal.
- `502 Bad Gateway` when an external LLM retry fails before producing usable
  text.

## Backend State And Artifacts

Extend the in-memory game record with a recoverable failed attempt reference:

- `pending_failure_rollout_id`
- `pending_failure_judgment_id`
- `pending_failure_reason`

This state is set when an LLM move has `legal == false` or lacks a parsed move.
It is cleared only when a legal recovery move advances the game or when a new
game is created.

Artifact metadata for failed attempts must include:

- `game_id`
- `source: "ui_live"`
- `requires_recovery: true`

Artifact metadata for recovery moves must include:

- `game_id`
- `source: "ui_live_recovery"`
- `recovery_action`
- `recovered_from_rollout_id`
- `recovered_from_judgment_id`

The existing `prompts.jsonl`, `rollouts.jsonl`, and `judgments.jsonl` files stay
the persistence format for V1.

`GameState` responses must expose recovery state so the frontend does not infer
it from judgment fields alone:

```json
{
  "pending_recovery": {
    "rollout_id": "rollout-...",
    "judgment_id": "judgment-...",
    "reason": "illegal_move"
  }
}
```

`pending_recovery` is `null` when no failed LLM attempt is waiting for explicit
user action.

## Recovery Action Semantics

### Retry LLM

Uses the same LLM client and normal prompt path as `llm-move`. The resulting
rollout is judged and persisted. Legal success advances the board. Illegal or
unparsed output keeps the game paused and replaces the visible failed attempt.

### Retry With Legal Moves

Uses the same LLM client but adds a stronger instruction that the answer must be
one of the current legal UCI moves. The prompt includes a sorted legal move
list. The result is judged and persisted like a normal retry.

### Use Teacher Move

Uses `last_judgment.teacher_move_uci` from the active failed attempt. The backend
must verify that the teacher move exists and is legal in the current board before
applying it. If unavailable or illegal, return `400 Bad Request` with a clear
message and leave the game paused.

### Use Deterministic Legal Move

Uses `DeterministicLegalMoveClient` against the current FEN, independent of the
configured LLM mode. The result is persisted as a rollout/judgment pair and,
when legal, applied to the board.

## Frontend UI

Add an `LLM Runtime` or `Recovery` panel in the inspector, near the existing
reasoning/judgment/tools sections.

Always show:

- LLM mode from `/api/health`.
- Model from `/api/health`.
- Last request status derived from the latest rollout/judgment.

When recovery is active, show:

- A clear paused state.
- Failed parsed move or parse error.
- Failure bucket and feedback.
- Recovery buttons:
  - `Retry LLM`
  - `Retry with legal moves`
  - `Use teacher move`
  - `Use deterministic legal move`
  - `Leave paused`

Button availability:

- `Use teacher move` is disabled unless a teacher move is present.
- Retry buttons are disabled while a request is in flight.
- `Leave paused` is a local UI no-op that keeps the failed attempt visible.

The board remains interactive only if it is still the human side's turn.
When the LLM side is paused for recovery, normal human move entry and board
dragging remain disabled.

## Timeline Behavior

V1 does not represent failed LLM attempts as played board moves. The timeline
continues to show only moves that actually advanced the board. The failed attempt
is visible in the inspector. A future move-attempt history can add separate
attempt chips.

## Error Handling

- External LLM call fails: show backend error, keep game paused.
- Retry returns illegal output: show new failed output, keep game paused.
- Teacher unavailable: disable the button and return a clear backend error if
  called directly.
- Legal stub finds no legal moves: return a clear backend error, keep game
  paused.
- Recovery endpoint called without pending failure: return a clear backend
  error.

## Testing

### Backend

- Illegal LLM output records rollout/judgment and leaves board FEN unchanged.
- Illegal LLM output sets pending recovery state in `GameState`.
- `retry` calls the configured LLM path and applies a legal retry.
- `retry_with_legal_moves` includes legal moves in the prompt and applies a
  legal retry.
- `teacher` applies a legal teacher move and rejects unavailable/illegal teacher
  moves.
- `legal_stub` applies a deterministic legal move independent of configured
  LLM mode.
- Recovery artifacts include recovery metadata.
- Calling recovery without pending failure returns `400`.

### Frontend

- Recovery panel appears for illegal or unparsed latest LLM judgment.
- Recovery panel does not appear for legal latest judgment.
- Buttons call `POST /api/games/{game_id}/recovery-move` with the correct
  actions.
- Successful recovery updates board, timeline, reasoning, and judgment.
- Teacher button disables when no teacher move exists.

### Browser Smoke

- Start app in static-illegal test mode.
- New game, play a human move, ask LLM.
- Verify board does not advance and recovery panel appears.
- Click `Use deterministic legal move`.
- Verify board advances, timeline adds the recovery move, and judgment is legal.

## Open Decisions

No open product decisions remain for V1. The user selected explicit recovery
instead of automatic fallback and approved the five recovery actions listed in
this spec.
