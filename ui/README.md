# Chess LLM Workbench UI

Research/debug UI for playing against a chess LLM and reviewing judged rollout
artifacts.

## Backend

Install the root package first so the backend can import `chess_llm`:

```bash
python -m pip install -e ".[ui]"
```

Run the API:

```bash
uvicorn chess_llm_ui.app:app --reload
```

Useful environment variables:

```bash
CHESS_UI_LLM_MODE=auto
CHESS_UI_LLM_BASE_URL=http://127.0.0.1:8001/v1
CHESS_UI_LLM_API_KEY=local-key
CHESS_UI_LLM_MODEL=Qwen/Qwen3-4B
CHESS_UI_ARTIFACT_ROOT=ui/runs
CHESS_UI_BOOK_ROOT=sft/make_data/polyglot_opening_books
CHESS_UI_STOCKFISH_PATH=C:/tools/stockfish/stockfish.exe
CHESS_UI_STOCKFISH_DEPTH=16
CHESS_UI_STOCKFISH_THREADS=1
CHESS_UI_STOCKFISH_HASH_MB=256
CHESS_UI_TOOL_BATCH_LIMIT=100
```

The LLM endpoint is expected to expose an OpenAI-compatible
`/chat/completions` route. `CHESS_UI_LLM_MODE=auto` uses that endpoint when
`CHESS_UI_LLM_BASE_URL` is set. Without a configured endpoint, `auto` falls
back to `legal_stub`, a deterministic local client that picks the first sorted
legal move for the current FEN and returns it as `<think>...</think><move>...</move>`.
Set `CHESS_UI_LLM_MODE=legal_stub` explicitly when you want to test the UI
without starting an LLM server.

## Stockfish And Tools

Stockfish is optional. When `CHESS_UI_STOCKFISH_PATH` is unset or points to a
missing binary, live play still works and LLM moves are judged with
legality-only feedback. When the path is valid, the backend lazily opens one
Stockfish process, serializes analysis through a lock, and adds teacher move,
regret, depth, and score metadata to live `judgments.jsonl` rows.

The inspector's Tools panel calls:

- `GET /api/tools` for Stockfish status and enabled capabilities.
- `POST /api/tools/analyze` for legal moves, SAN/capture/check metadata,
  opening-book suggestions, and optional Stockfish best move/eval/PV.
- `POST /api/artifacts/{run_id}/score` for manual review scoring.

Manual review scoring is in-memory for v1. `Score selected` updates the active
rollout row, and `Score visible` sends the currently filtered rollout ids up to
`CHESS_UI_TOOL_BATCH_LIMIT`. The source artifact directory is not rewritten.

## Frontend

```bash
cd ui
npm install
npm run dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8000`.

## Artifact Review

Review mode loads a local directory containing the files written by
`chess-llm-batch-judge`:

- `prompts.jsonl`
- `rollouts.jsonl`
- `judgments.jsonl`

The UI joins those files by `prompt_id` and `rollout_id`, then lets you filter
by model, legal status, failure bucket, split, regret, and free text.

## Checks

```bash
python -m pytest tests/test_opening_books.py tests/test_inference_contracts.py tests/test_ui_backend.py -q
cd ui
npm test
npm run build
```
