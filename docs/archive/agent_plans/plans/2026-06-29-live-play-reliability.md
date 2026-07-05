# Live Play Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit live-game recovery loop for illegal or unparsed LLM moves.

**Architecture:** Keep recovery state on the backend `GameRecord` and expose it through `GameState.pending_recovery`. Reuse the existing prompt, rollout, judgment, and artifact paths, then add a single recovery endpoint that can retry the configured LLM, retry with legal moves emphasized, apply the Stockfish teacher move, or apply the deterministic legal stub. The frontend reads `pending_recovery` and renders a focused recovery panel in the inspector.

**Tech Stack:** FastAPI, python-chess, existing `chess_llm` artifact dataclasses, React/Vite TypeScript, Vitest, Testing Library.

---

## File Structure

- Modify `src/chess_llm_ui/app.py`
  - Add recovery request model and pending recovery dataclass/state.
  - Factor LLM attempt handling so normal moves and recovery retries share one path.
  - Add `POST /api/games/{game_id}/recovery-move`.
  - Expose `pending_recovery` in `_game_to_dict`.
- Modify `tests/test_ui_backend.py`
  - Add test LLM clients for illegal then legal retry behavior.
  - Add backend tests for paused illegal attempts and each recovery action.
- Modify `ui/src/domain/types.ts`
  - Add `PendingRecovery` and `RecoveryAction`.
  - Add `pending_recovery` to `GameState`.
- Modify `ui/src/api.ts`
  - Add `recoverGameMove(gameId, action)`.
- Modify `ui/src/App.tsx`
  - Add recovery action handler.
  - Add `LLMRuntimePanel`/recovery panel in the inspector.
  - Disable normal live controls while LLM-side recovery is pending.
- Modify `ui/src/App.test.tsx`
  - Add frontend tests for panel rendering, disabled teacher state, and recovery button calls.

---

### Task 1: Backend Pending Recovery State

**Files:**
- Modify: `tests/test_ui_backend.py`
- Modify: `src/chess_llm_ui/app.py`

- [ ] **Step 1: Write failing backend test for illegal LLM pause**

Add a test client that returns an illegal repeated move after a human move:

```python
def test_illegal_llm_move_sets_pending_recovery_and_does_not_advance_board(tmp_path):
    settings = BackendSettings(
        artifact_root=tmp_path / "runs",
        book_root=tmp_path / "books",
        llm_model="unit-model",
    )
    app = create_app(
        settings=settings,
        llm_client=StaticLLMClient("<think>Repeat center.</think>\n<move>e2e4</move>"),
    )
    local_client = TestClient(app)
    created = local_client.post("/api/games", json={"human_side": "black"})
    game_id = created.json()["game_id"]
    before_fen = created.json()["fen"]

    response = local_client.post(f"/api/games/{game_id}/llm-move")

    assert response.status_code == 200
    payload = response.json()
    assert payload["fen"] == before_fen
    assert payload["moves"] == []
    assert payload["last_judgment"]["legal"] is False
    assert payload["pending_recovery"]["rollout_id"] == payload["last_rollout"]["rollout_id"]
    assert payload["pending_recovery"]["judgment_id"] == payload["last_judgment"]["judgment_id"]
    assert payload["pending_recovery"]["reason"] == "illegal_move"
    assert payload["last_judgment"]["metadata"]["requires_recovery"] is True
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python -m pytest tests/test_ui_backend.py::test_illegal_llm_move_sets_pending_recovery_and_does_not_advance_board -q
```

Expected: fail because `pending_recovery` is missing.

- [ ] **Step 3: Implement minimal pending recovery state**

In `src/chess_llm_ui/app.py` add:

```python
class RecoveryMoveRequest(BaseModel):
    action: str = Field(pattern="^(retry|retry_with_legal_moves|teacher|legal_stub)$")


@dataclass
class PendingRecovery:
    rollout_id: str
    judgment_id: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "rollout_id": self.rollout_id,
            "judgment_id": self.judgment_id,
            "reason": self.reason,
        }
```

Add `pending_recovery: PendingRecovery | None = None` to `GameRecord`.

After judging an LLM rollout, set pending state when no legal move is applied:

```python
if judgment.legal is True and rollout.parsed_answer.move_uci is not None:
    ...
    record.pending_recovery = None
else:
    judgment.metadata["requires_recovery"] = True
    record.pending_recovery = PendingRecovery(
        rollout_id=rollout.rollout_id,
        judgment_id=judgment.judgment_id,
        reason=judgment.failure_bucket or rollout.parsed_answer.parse_error or "unparsed_move",
    )
```

Expose it in `_game_to_dict`:

```python
"pending_recovery": record.pending_recovery.to_dict() if record.pending_recovery else None,
```

- [ ] **Step 4: Run backend test and verify GREEN**

Run:

```bash
python -m pytest tests/test_ui_backend.py::test_illegal_llm_move_sets_pending_recovery_and_does_not_advance_board -q
```

Expected: pass.

---

### Task 2: Backend Recovery Endpoint

**Files:**
- Modify: `tests/test_ui_backend.py`
- Modify: `src/chess_llm_ui/app.py`

- [ ] **Step 1: Write failing recovery tests**

Add tests for:

```python
class SequenceLLMClient:
    def __init__(self, raw_outputs: list[str]) -> None:
        self.raw_outputs = list(raw_outputs)
        self.requests: list[InferenceRequest] = []

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        raw_output = self.raw_outputs.pop(0)
        return InferenceResponse(
            model_id=request.model_id,
            raw_text=raw_output,
            metadata={"client": "sequence-test"},
        )
```

Then add:

```python
def test_recovery_retry_applies_legal_retry_and_records_metadata(tmp_path):
    llm = SequenceLLMClient([
        "<think>Bad.</think>\n<move>e2e4</move>",
        "<think>Try knight.</think>\n<move>g1f3</move>",
    ])
    settings = BackendSettings(artifact_root=tmp_path / "runs", book_root=tmp_path / "books", llm_model="unit-model")
    app = create_app(settings=settings, llm_client=llm)
    local_client = TestClient(app)
    game_id = local_client.post("/api/games", json={"human_side": "black"}).json()["game_id"]
    failed = local_client.post(f"/api/games/{game_id}/llm-move").json()

    response = local_client.post(f"/api/games/{game_id}/recovery-move", json={"action": "retry"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["pending_recovery"] is None
    assert payload["moves"][-1]["move_uci"] == "g1f3"
    assert payload["moves"][-1]["rollout"]["metadata"]["recovery_action"] == "retry"
    assert payload["moves"][-1]["rollout"]["metadata"]["recovered_from_rollout_id"] == failed["last_rollout"]["rollout_id"]
```

Add:

```python
def test_recovery_retry_with_legal_moves_adds_legal_move_list_to_prompt(tmp_path):
    llm = SequenceLLMClient([
        "<think>Bad.</think>\n<move>e2e4</move>",
        "<think>Use listed move.</think>\n<move>g1f3</move>",
    ])
    settings = BackendSettings(artifact_root=tmp_path / "runs", book_root=tmp_path / "books", llm_model="unit-model")
    app = create_app(settings=settings, llm_client=llm)
    local_client = TestClient(app)
    game_id = local_client.post("/api/games", json={"human_side": "black"}).json()["game_id"]
    local_client.post(f"/api/games/{game_id}/llm-move")

    response = local_client.post(f"/api/games/{game_id}/recovery-move", json={"action": "retry_with_legal_moves"})

    assert response.status_code == 200
    assert "Legal UCI moves:" in llm.requests[-1].messages[-1].content
```

Add:

```python
def test_recovery_legal_stub_applies_deterministic_move_after_static_failure(tmp_path):
    settings = BackendSettings(artifact_root=tmp_path / "runs", book_root=tmp_path / "books", llm_model="unit-model")
    app = create_app(settings=settings, llm_client=StaticLLMClient("<think>Bad.</think>\n<move>e2e4</move>"))
    local_client = TestClient(app)
    game_id = local_client.post("/api/games", json={"human_side": "black"}).json()["game_id"]
    local_client.post(f"/api/games/{game_id}/llm-move")

    response = local_client.post(f"/api/games/{game_id}/recovery-move", json={"action": "legal_stub"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["pending_recovery"] is None
    assert payload["moves"][-1]["source"] == "llm"
    assert payload["moves"][-1]["judgment"]["legal"] is True
    assert payload["moves"][-1]["rollout"]["metadata"]["recovery_action"] == "legal_stub"
```

Add:

```python
def test_recovery_without_pending_failure_returns_400(client):
    game_id = client.post("/api/games", json={"human_side": "white"}).json()["game_id"]

    response = client.post(f"/api/games/{game_id}/recovery-move", json={"action": "legal_stub"})

    assert response.status_code == 400
    assert "no pending recovery" in response.json()["detail"].lower()
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest tests/test_ui_backend.py -q
```

Expected: fail because `/api/games/{game_id}/recovery-move` does not exist.

- [ ] **Step 3: Implement shared LLM attempt helper and endpoint**

Extract the current `llm_move` body into helper functions:

```python
def _run_llm_attempt(
    *,
    app: FastAPI,
    record: GameRecord,
    settings: BackendSettings,
    client: InferenceClient,
    model_side: str,
    prompt_variant: str = "normal",
    recovery_metadata: dict | None = None,
) -> None:
    prompt = _build_prompt(
        record,
        settings.llm_model,
        include_legal_moves=prompt_variant == "legal_moves",
    )
    response = client.complete(...)
    metadata = {"game_id": record.game_id, **response.metadata}
    if recovery_metadata:
        metadata.update(recovery_metadata)
    rollout = build_rollout(prompt, response.model_id, response.raw_text, metadata=metadata)
    judgment = app.state.tool_service.judge_rollout(
        prompt,
        rollout,
        metadata={"game_id": record.game_id, "source": metadata.get("source", "ui_live")},
    )
    _record_llm_attempt(record, rollout, judgment, model_side, recovery_metadata=recovery_metadata)
```

Implement `_record_llm_attempt(...)` to:
- set `last_rollout` and `last_judgment`
- push legal move and clear pending recovery
- otherwise set `requires_recovery` metadata and pending recovery
- append prompt/rollout/judgment JSONL

Change `_build_prompt` signature:

```python
def _build_prompt(record: GameRecord, model_id: str, *, include_legal_moves: bool = False) -> PromptArtifact:
```

When `include_legal_moves` is true, append:

```python
f"Legal UCI moves: {' '.join(sorted(move.uci() for move in record.board.legal_moves))}\n"
```

Add endpoint:

```python
@app.post("/api/games/{game_id}/recovery-move")
def recovery_move(game_id: str, request_payload: RecoveryMoveRequest) -> dict:
    with _get_game_lock(app, game_id):
        record = _get_game(app, game_id)
        if record.pending_recovery is None:
            raise HTTPException(status_code=400, detail="No pending recovery for this game.")
        model_side = "black" if record.human_side == "white" else "white"
        _ensure_turn(record, model_side)
        recovery_metadata = {
            "source": "ui_live_recovery",
            "recovery_action": request_payload.action,
            "recovered_from_rollout_id": record.pending_recovery.rollout_id,
            "recovered_from_judgment_id": record.pending_recovery.judgment_id,
        }
        if request_payload.action == "legal_stub":
            client = DeterministicLegalMoveClient(model_id=resolved_settings.llm_model)
            _run_llm_attempt(..., client=client, recovery_metadata=recovery_metadata)
        elif request_payload.action == "retry":
            _run_llm_attempt(..., client=app.state.llm_client, recovery_metadata=recovery_metadata)
        elif request_payload.action == "retry_with_legal_moves":
            _run_llm_attempt(..., client=app.state.llm_client, prompt_variant="legal_moves", recovery_metadata=recovery_metadata)
        elif request_payload.action == "teacher":
            _apply_teacher_recovery(...)
        return _game_to_dict(record)
```

Implement teacher recovery by validating `record.last_judgment.teacher_move_uci`, creating a synthetic rollout raw output with `<move>{teacher}</move>`, judging it, applying it through `_record_llm_attempt`, and preserving recovery metadata.

- [ ] **Step 4: Run backend tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_ui_backend.py -q
```

Expected: all `test_ui_backend.py` tests pass.

---

### Task 3: Frontend API And Types

**Files:**
- Modify: `ui/src/domain/types.ts`
- Modify: `ui/src/api.ts`
- Modify: `ui/src/App.test.tsx`

- [ ] **Step 1: Write failing frontend API/UI test**

Update the API mock in `ui/src/App.test.tsx` to include `recoverGameMove`.

Add a test:

```tsx
it("shows recovery controls for a pending illegal LLM move and calls legal stub recovery", async () => {
  vi.mocked(createGame).mockResolvedValue(recoveryGame("recovery-game"));
  vi.mocked(recoverGameMove).mockResolvedValue(gameState("recovered-game"));

  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: "New game" }));
  expect(await screen.findByText("Recovery paused")).toBeInTheDocument();
  expect(screen.getByText("illegal_move")).toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "Use deterministic legal move" }));

  await waitFor(() => {
    expect(recoverGameMove).toHaveBeenCalledWith("recovery-game", "legal_stub");
  });
});
```

Add `recoveryGame(gameId)` helper returning a `GameState` with:

```ts
pending_recovery: {
  rollout_id: "rollout-bad",
  judgment_id: "judgment-bad",
  reason: "illegal_move"
}
```

and `last_rollout` / `last_judgment` showing an illegal `e2e4` response.

- [ ] **Step 2: Run frontend test and verify RED**

Run:

```bash
cd ui
npm test -- --run src/App.test.tsx -t "recovery controls"
```

Expected: fail because `recoverGameMove` and recovery UI are missing.

- [ ] **Step 3: Add types and API function**

In `types.ts` add:

```ts
export type RecoveryAction = "retry" | "retry_with_legal_moves" | "teacher" | "legal_stub";

export interface PendingRecovery {
  rollout_id: string;
  judgment_id: string;
  reason: string;
}
```

Add to `GameState`:

```ts
pending_recovery: PendingRecovery | null;
```

In `api.ts` import `RecoveryAction` and add:

```ts
export function recoverGameMove(gameId: string, action: RecoveryAction): Promise<GameState> {
  return requestJson(`/api/games/${gameId}/recovery-move`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ action })
  });
}
```

- [ ] **Step 4: Run TypeScript-targeted test**

Run:

```bash
cd ui
npm test -- --run src/App.test.tsx -t "recovery controls"
```

Expected: still fail until UI is added, but TypeScript imports resolve.

---

### Task 4: Frontend Recovery Panel

**Files:**
- Modify: `ui/src/App.tsx`
- Modify: `ui/src/App.test.tsx`
- Modify: `ui/src/styles.css`

- [ ] **Step 1: Implement recovery handler and panel**

In `App.tsx` import `recoverGameMove` and `RecoveryAction`.

Add:

```ts
const hasPendingRecovery = mode === "play" && Boolean(game?.pending_recovery);
const liveControlsDisabled = !game || busy || isInspectingHistoricalPlayMove || hasPendingRecovery;
```

Add handler:

```ts
async function handleRecoveryAction(action: RecoveryAction) {
  if (!game || !game.pending_recovery) {
    return;
  }
  await runAction(async () => {
    applyGameState(await recoverGameMove(game.game_id, action));
  });
}
```

Render a `RecoveryPanel` in the inspector when `mode === "play"`:

```tsx
<RecoveryPanel
  game={game}
  health={health}
  busy={busy}
  onRecover={handleRecoveryAction}
/>
```

`RecoveryPanel` shows runtime mode/model always and shows action buttons only when `game?.pending_recovery` exists. Disable `Use teacher move` if `game.last_judgment?.teacher_move_uci` is missing.

- [ ] **Step 2: Add styling**

In `styles.css`, add compact `.runtime-panel`, `.recovery-actions`, and `.warning-state` classes matching existing inspector panels.

- [ ] **Step 3: Run focused frontend tests**

Run:

```bash
cd ui
npm test -- --run src/App.test.tsx
```

Expected: all App tests pass.

---

### Task 5: Verification And Browser Smoke

**Files:**
- Runtime only unless a bug is found.

- [ ] **Step 1: Run targeted backend tests**

Run:

```bash
python -m pytest tests/test_ui_backend.py tests/test_stockfish_judge.py tests/test_external_stockfish.py -q
```

Expected: all pass.

- [ ] **Step 2: Run frontend tests and build**

Run:

```bash
cd ui
npm test -- --run
npm run build
```

Expected: all tests pass and Vite build succeeds.

- [ ] **Step 3: Browser smoke**

Run app with a static illegal LLM response or use a test backend. In the browser:

1. Create a game as black so the LLM moves first.
2. Ask LLM with a static illegal `e2e4`.
3. Verify board FEN does not advance and recovery panel appears.
4. Click `Use deterministic legal move`.
5. Verify timeline adds one LLM move, `pending_recovery` clears, and judgment is legal.

Expected: no relevant console errors or framework overlays.
