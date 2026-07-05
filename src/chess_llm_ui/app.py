"""FastAPI app for the Chess LLM workbench."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import threading
from pathlib import Path
from uuid import uuid4

import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chess_llm.artifacts.schemas import ChatMessage, JudgmentArtifact, PromptArtifact, RolloutArtifact
from chess_llm.autodata.rollouts import build_rollout
from chess_llm.core.opening_books import (
    OpeningBook,
    discover_opening_books,
    sample_polyglot_book_line,
)
from chess_llm.formats.prompts import SYSTEM_PROMPT
from chess_llm.inference import InferenceClient, InferenceRequest, InferenceResponse

from chess_llm_ui.artifact_review import ArtifactRun
from chess_llm_ui.artifact_routes import register_artifact_routes
from chess_llm_ui.config import BackendSettings
from chess_llm_ui.llm import (
    DeterministicLegalMoveClient,
    OpenAICompatibleLLMClient,
    StaticLLMClient,
)
from chess_llm_ui.tools import ToolService


class CreateGameRequest(BaseModel):
    human_side: str = Field(default="white", pattern="^(white|black)$")
    book_id: str | None = None
    book_max_plies: int = 0
    seed: int | None = None


class MoveRequest(BaseModel):
    move_uci: str


class RecoveryMoveRequest(BaseModel):
    action: str = Field(pattern="^(retry|retry_with_legal_moves|teacher|legal_stub)$")


class AnalyzePositionRequest(BaseModel):
    fen: str
    book_id: str | None = None
    include_stockfish: bool = False
    depth: int | None = Field(default=None, ge=1)


@dataclass
class PendingRecovery:
    """A failed live model attempt that needs an explicit user recovery action."""

    rollout_id: str
    judgment_id: str
    reason: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "rollout_id": self.rollout_id,
            "judgment_id": self.judgment_id,
            "reason": self.reason,
            "created_at": self.created_at,
        }


@dataclass
class GameRecord:
    """In-memory state for one live game."""

    game_id: str
    board: chess.Board
    human_side: str
    artifact_dir: Path
    moves: list[dict] = field(default_factory=list)
    book_line: list[dict] = field(default_factory=list)
    last_rollout: RolloutArtifact | None = None
    last_judgment: JudgmentArtifact | None = None
    pending_recovery: PendingRecovery | None = None
    prompt_counter: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def create_app(
    *,
    settings: BackendSettings | None = None,
    llm_client: InferenceClient | None = None,
    tool_service: ToolService | None = None,
) -> FastAPI:
    """Create the FastAPI application."""
    resolved_settings = settings or BackendSettings.from_env()
    resolved_settings.artifact_root.mkdir(parents=True, exist_ok=True)
    llm_mode = "injected"
    if llm_client is None:
        llm_mode = _resolve_llm_mode(resolved_settings)
        llm_client = _build_default_llm_client(resolved_settings, llm_mode)

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        try:
            yield
        finally:
            fastapi_app.state.tool_service.close()

    app = FastAPI(title="Chess LLM Workbench API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = resolved_settings
    app.state.llm_client = llm_client
    app.state.llm_mode = llm_mode
    app.state.tool_service = tool_service or ToolService(settings=resolved_settings)
    app.state.games: dict[str, GameRecord] = {}
    app.state.game_locks: dict[str, threading.RLock] = {}
    app.state.games_lock = threading.Lock()
    app.state.artifact_runs: dict[str, ArtifactRun] = {}
    app.state.artifact_runs_lock = threading.Lock()
    register_artifact_routes(app, resolved_settings)

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": resolved_settings.llm_model,
            "llm_mode": app.state.llm_mode,
        }

    @app.get("/api/books")
    def books() -> dict:
        discovered = discover_opening_books(resolved_settings.book_root)
        return {"items": [_book_to_dict(book) for book in discovered]}

    @app.get("/api/tools")
    def tools_status() -> dict:
        return app.state.tool_service.status()

    @app.post("/api/tools/analyze")
    def analyze_tools(request_payload: AnalyzePositionRequest) -> dict:
        try:
            return app.state.tool_service.analyze_position(
                fen=request_payload.fen,
                book_id=request_payload.book_id,
                include_stockfish=request_payload.include_stockfish,
                depth=request_payload.depth,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/games")
    def create_game(request_payload: CreateGameRequest) -> dict:
        game_id = f"game-{uuid4().hex[:12]}"
        board = chess.Board()
        artifact_dir = resolved_settings.artifact_root / game_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        record = GameRecord(
            game_id=game_id,
            board=board,
            human_side=request_payload.human_side,
            artifact_dir=artifact_dir,
        )
        if request_payload.book_id and request_payload.book_max_plies > 0:
            _apply_book_line(record, request_payload, resolved_settings.book_root)
        with app.state.games_lock:
            app.state.games[game_id] = record
            app.state.game_locks[game_id] = threading.RLock()
        return _game_to_dict(record)

    @app.get("/api/games/{game_id}")
    def get_game(game_id: str) -> dict:
        with _get_game_lock(app, game_id):
            return _game_to_dict(_get_game(app, game_id))

    @app.post("/api/games/{game_id}/human-move")
    def human_move(game_id: str, request_payload: MoveRequest) -> dict:
        with _get_game_lock(app, game_id):
            record = _get_game(app, game_id)
            _ensure_turn(record, record.human_side)
            move = _parse_legal_move(record.board, request_payload.move_uci)
            side = _side_for_turn(record.board.turn)
            record.moves.append(
                {
                    "source": "human",
                    "side": side,
                    "move_uci": move.uci(),
                    "fen_before": record.board.fen(),
                }
            )
            record.board.push(move)
            return _game_to_dict(record)

    @app.post("/api/games/{game_id}/llm-move")
    def llm_move(game_id: str) -> dict:
        with _get_game_lock(app, game_id):
            record = _get_game(app, game_id)
            if record.pending_recovery is not None:
                raise HTTPException(
                    status_code=400,
                    detail="game has a pending recovery; choose a recovery action",
                )
            model_side = "black" if record.human_side == "white" else "white"
            _ensure_turn(record, model_side)
            try:
                prompt, rollout, judgment = _run_llm_attempt(
                    app,
                    record,
                    resolved_settings,
                    app.state.llm_client,
                    source="ui_live",
                )
            except RuntimeError as exc:
                raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc
            _apply_llm_attempt(record, model_side, rollout, judgment)
            _append_live_artifacts(record, prompt, rollout, judgment)
            return _game_to_dict(record)

    @app.post("/api/games/{game_id}/recovery-move")
    def recovery_move(game_id: str, request_payload: RecoveryMoveRequest) -> dict:
        with _get_game_lock(app, game_id):
            record = _get_game(app, game_id)
            pending = record.pending_recovery
            if pending is None:
                raise HTTPException(status_code=400, detail="no pending recovery for this game")

            model_side = "black" if record.human_side == "white" else "white"
            _ensure_turn(record, model_side)
            recovery_metadata = {
                "recovery_action": request_payload.action,
                "recovered_from_rollout_id": pending.rollout_id,
                "recovered_from_judgment_id": pending.judgment_id,
            }

            try:
                if request_payload.action == "teacher":
                    prompt, rollout, judgment = _teacher_recovery_attempt(
                        app,
                        record,
                        resolved_settings,
                        recovery_metadata=recovery_metadata,
                    )
                else:
                    client = app.state.llm_client
                    if request_payload.action == "legal_stub":
                        client = DeterministicLegalMoveClient(model_id=resolved_settings.llm_model)
                    prompt, rollout, judgment = _run_llm_attempt(
                        app,
                        record,
                        resolved_settings,
                        client,
                        source="ui_live_recovery",
                        include_legal_moves=(
                            request_payload.action
                            in {"retry_with_legal_moves", "legal_stub"}
                        ),
                        extra_metadata=recovery_metadata,
                    )
            except RuntimeError as exc:
                raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

            _apply_llm_attempt(record, model_side, rollout, judgment)
            _append_live_artifacts(record, prompt, rollout, judgment)
            return _game_to_dict(record)

    return app


def _resolve_llm_mode(settings: BackendSettings) -> str:
    mode = settings.llm_mode.strip().lower()
    if mode == "auto":
        return "openai" if settings.llm_base_url else "legal_stub"
    if mode not in {"openai", "legal_stub", "static"}:
        raise ValueError(
            "CHESS_UI_LLM_MODE must be one of: auto, openai, legal_stub, static"
        )
    return mode


def _build_default_llm_client(settings: BackendSettings, llm_mode: str) -> InferenceClient:
    if llm_mode == "openai":
        if not settings.llm_base_url:
            raise ValueError("CHESS_UI_LLM_BASE_URL is required when CHESS_UI_LLM_MODE=openai")
        return OpenAICompatibleLLMClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
    if llm_mode == "legal_stub":
        return DeterministicLegalMoveClient(model_id=settings.llm_model)
    if llm_mode == "static":
        return StaticLLMClient(
            "<think>Static test response.</think>\n<move>e2e4</move>",
            model_id=settings.llm_model,
        )
    raise ValueError(f"unsupported LLM mode: {llm_mode}")


def _apply_book_line(
    record: GameRecord,
    request_payload: CreateGameRequest,
    book_root: Path,
) -> None:
    books = discover_opening_books(book_root)
    matching = [book for book in books if book.book_id == request_payload.book_id]
    if not matching:
        return
    line = sample_polyglot_book_line(
        matching[0].path,
        max_plies=request_payload.book_max_plies,
        seed=request_payload.seed,
    )
    for sampled in line.moves:
        move = chess.Move.from_uci(sampled.move_uci)
        record.book_line.append(
            {
                "source": "book",
                "move_uci": sampled.move_uci,
                "weight": sampled.weight,
                "fen_before": sampled.fen_before,
            }
        )
        record.moves.append(record.book_line[-1])
        record.board.push(move)


def _run_llm_attempt(
    app: FastAPI,
    record: GameRecord,
    settings: BackendSettings,
    llm_client: InferenceClient,
    *,
    source: str,
    include_legal_moves: bool = False,
    extra_metadata: dict | None = None,
) -> tuple[PromptArtifact, RolloutArtifact, JudgmentArtifact]:
    prompt = _build_prompt(
        record,
        settings.llm_model,
        include_legal_moves=include_legal_moves,
        source=source,
        extra_metadata=extra_metadata,
    )
    response = llm_client.complete(
        InferenceRequest(
            model_id=settings.llm_model,
            messages=prompt.messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            metadata={"game_id": record.game_id, "fen": record.board.fen()},
        )
    )
    rollout, judgment = _build_and_judge_response(
        app,
        record,
        prompt,
        response,
        source=source,
        extra_metadata=extra_metadata,
    )
    return prompt, rollout, judgment


def _teacher_recovery_attempt(
    app: FastAPI,
    record: GameRecord,
    settings: BackendSettings,
    *,
    recovery_metadata: dict,
) -> tuple[PromptArtifact, RolloutArtifact, JudgmentArtifact]:
    teacher_move = record.last_judgment.teacher_move_uci if record.last_judgment else None
    if not teacher_move:
        raise HTTPException(
            status_code=400,
            detail="pending recovery does not have a teacher move",
        )
    _parse_legal_move(record.board, teacher_move)
    prompt = _build_prompt(
        record,
        settings.llm_model,
        include_legal_moves=True,
        source="ui_live_recovery",
        extra_metadata=recovery_metadata,
    )
    response = InferenceResponse(
        model_id=settings.llm_model,
        raw_text=(
            "<think>Teacher move selected for live recovery.</think>\n"
            f"<move>{teacher_move}</move>"
        ),
        metadata={"client": "teacher_recovery", "fen": record.board.fen()},
    )
    rollout, judgment = _build_and_judge_response(
        app,
        record,
        prompt,
        response,
        source="ui_live_recovery",
        extra_metadata=recovery_metadata,
    )
    return prompt, rollout, judgment


def _build_and_judge_response(
    app: FastAPI,
    record: GameRecord,
    prompt: PromptArtifact,
    response: InferenceResponse,
    *,
    source: str,
    extra_metadata: dict | None = None,
) -> tuple[RolloutArtifact, JudgmentArtifact]:
    resolved_extra_metadata = dict(extra_metadata or {})
    rollout = build_rollout(
        prompt,
        response.model_id,
        response.raw_text,
        metadata={
            "game_id": record.game_id,
            "source": source,
            **response.metadata,
            **resolved_extra_metadata,
        },
    )
    judgment = app.state.tool_service.judge_rollout(
        prompt,
        rollout,
        metadata={
            "game_id": record.game_id,
            "source": source,
            **resolved_extra_metadata,
        },
    )
    if judgment.legal is not True:
        _populate_teacher_move_if_available(app, record, judgment)
    return rollout, judgment


def _apply_llm_attempt(
    record: GameRecord,
    model_side: str,
    rollout: RolloutArtifact,
    judgment: JudgmentArtifact,
) -> None:
    record.last_rollout = rollout
    record.last_judgment = judgment

    if judgment.legal is True and rollout.parsed_answer.move_uci is not None:
        move = chess.Move.from_uci(rollout.parsed_answer.move_uci)
        if move not in record.board.legal_moves:
            judgment.legal = False
            judgment.failure_bucket = "illegal_move"
            judgment.feedback = (
                f"Parsed move {rollout.parsed_answer.move_uci} is illegal "
                "in the current game position."
            )
        else:
            record.pending_recovery = None
            record.moves.append(
                {
                    "source": "llm",
                    "side": model_side,
                    "move_uci": move.uci(),
                    "fen_before": record.board.fen(),
                    "rollout_id": rollout.rollout_id,
                    "judgment_id": judgment.judgment_id,
                    "rollout": rollout.to_dict(),
                    "judgment": judgment.to_dict(),
                }
            )
            record.board.push(move)
            return

    rollout.metadata["requires_recovery"] = True
    judgment.metadata["requires_recovery"] = True
    reason = (
        judgment.feedback
        or rollout.parsed_answer.parse_error
        or "Model did not produce a legal move."
    )
    record.pending_recovery = PendingRecovery(
        rollout_id=rollout.rollout_id,
        judgment_id=judgment.judgment_id,
        reason=reason,
    )


def _populate_teacher_move_if_available(
    app: FastAPI,
    record: GameRecord,
    judgment: JudgmentArtifact,
) -> None:
    if judgment.teacher_move_uci:
        return
    try:
        analysis = app.state.tool_service.analyze_position(
            fen=record.board.fen(),
            include_stockfish=True,
        )
    except Exception as exc:
        judgment.metadata["teacher_lookup_error"] = str(exc)
        return
    stockfish = analysis.get("stockfish") or {}
    teacher_move = stockfish.get("best_move")
    if not teacher_move:
        return
    try:
        move = chess.Move.from_uci(teacher_move)
    except ValueError:
        return
    if move not in record.board.legal_moves:
        return
    judgment.teacher_move_uci = teacher_move
    judgment.metadata["teacher_source"] = "stockfish_analysis"
    judgment.metadata["teacher_depth"] = stockfish.get("depth")
    if stockfish.get("cp") is not None:
        judgment.metadata["teacher_eval_cp"] = stockfish.get("cp")


def _append_live_artifacts(
    record: GameRecord,
    prompt: PromptArtifact,
    rollout: RolloutArtifact,
    judgment: JudgmentArtifact,
) -> None:
    _append_artifact(record.artifact_dir / "prompts.jsonl", prompt.to_dict())
    _append_artifact(record.artifact_dir / "rollouts.jsonl", rollout.to_dict())
    _append_artifact(record.artifact_dir / "judgments.jsonl", judgment.to_dict())


def _build_prompt(
    record: GameRecord,
    model_id: str,
    *,
    include_legal_moves: bool = False,
    source: str = "ui_live",
    extra_metadata: dict | None = None,
) -> PromptArtifact:
    del model_id
    move_history = " ".join(move["move_uci"] for move in record.moves) or "(none)"
    lines = [
        "Choose the next legal chess move for the side to move.",
        f"FEN: {record.board.fen()}",
        f"Moves played: {move_history}",
    ]
    if include_legal_moves:
        legal_moves = ", ".join(sorted(move.uci() for move in record.board.legal_moves))
        lines.append(f"Legal UCI moves: {legal_moves}")
        lines.append("Choose exactly one move from the legal UCI moves list.")
    lines.append("Return exactly <think>short visible reasoning</think><move>UCI</move>.")
    content = "\n".join(lines)
    prompt_id = f"prompt-{record.game_id}-{record.prompt_counter:04d}"
    record.prompt_counter += 1
    return PromptArtifact(
        prompt_id=prompt_id,
        messages=[
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=content),
        ],
        fen=record.board.fen(),
        task_type="best_move",
        metadata={"game_id": record.game_id, "source": source, **dict(extra_metadata or {})},
    )


def _book_to_dict(book: OpeningBook) -> dict:
    return {
        "book_id": book.book_id,
        "name": book.name,
        "path": str(book.path),
        "curated": book.curated,
    }


def _get_game(app: FastAPI, game_id: str) -> GameRecord:
    record = app.state.games.get(game_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown game_id: {game_id}")
    return record


def _get_game_lock(app: FastAPI, game_id: str) -> threading.RLock:
    with app.state.games_lock:
        if game_id not in app.state.games:
            raise HTTPException(status_code=404, detail=f"unknown game_id: {game_id}")
        return app.state.game_locks.setdefault(game_id, threading.RLock())


def _ensure_turn(record: GameRecord, side: str) -> None:
    if _side_for_turn(record.board.turn) != side:
        raise HTTPException(status_code=400, detail=f"it is not {side}'s turn")


def _parse_legal_move(board: chess.Board, move_uci: str) -> chess.Move:
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid move UCI: {move_uci}") from exc
    if move not in board.legal_moves:
        raise HTTPException(status_code=400, detail=f"illegal move: {move_uci}")
    return move


def _side_for_turn(turn: chess.Color) -> str:
    return "white" if turn == chess.WHITE else "black"


def _game_to_dict(record: GameRecord) -> dict:
    return {
        "game_id": record.game_id,
        "human_side": record.human_side,
        "turn": _side_for_turn(record.board.turn),
        "fen": record.board.fen(),
        "created_at": record.created_at,
        "moves": list(record.moves),
        "book_line": list(record.book_line),
        "artifact_dir": str(record.artifact_dir),
        "last_rollout": record.last_rollout.to_dict() if record.last_rollout else None,
        "last_judgment": record.last_judgment.to_dict() if record.last_judgment else None,
        "pending_recovery": (
            record.pending_recovery.to_dict() if record.pending_recovery else None
        ),
        "legal_moves": sorted(move.uci() for move in record.board.legal_moves),
    }


def _append_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


app = create_app()
