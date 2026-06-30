import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import pytest
from fastapi.testclient import TestClient

from chess_llm.artifacts.jsonl import write_jsonl
from chess_llm.artifacts.schemas import (
    ChatMessage,
    JudgmentArtifact,
    ParsedAnswer,
    PromptArtifact,
    RolloutArtifact,
)
from chess_llm.inference import InferenceRequest, InferenceResponse

from chess_llm_ui.app import create_app
from chess_llm_ui.config import BackendSettings
from chess_llm_ui.llm import StaticLLMClient
from chess_llm_ui.tools import ToolService


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class FailingLLMClient:
    def complete(self, request: InferenceRequest) -> InferenceResponse:
        raise RuntimeError("endpoint down")


class BlockingLegalLLMClient:
    def __init__(self) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        with self._lock:
            self.calls += 1
            self.entered.set()
        self.release.wait(timeout=2.0)
        board = chess.Board(str(request.metadata["fen"]))
        move_uci = sorted(move.uci() for move in board.legal_moves)[0]
        return InferenceResponse(
            model_id=request.model_id,
            raw_text=f"<think>Blocked legal test move.</think>\n<move>{move_uci}</move>",
            metadata={"client": "blocking-test"},
        )


class SequenceLLMClient:
    def __init__(self, raw_texts: list[str]) -> None:
        self.raw_texts = raw_texts
        self.requests: list[InferenceRequest] = []

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.raw_texts) - 1)
        return InferenceResponse(
            model_id=request.model_id,
            raw_text=self.raw_texts[index],
            metadata={"client": "sequence-test", "call_index": index},
        )


class FakeStockfish:
    id = {"name": "Fake Stockfish"}

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, int | None]] = []
        self.configured: list[dict[str, int | str]] = []
        self.quit_called = False

    def configure(self, options: dict[str, int | str]) -> None:
        self.configured.append(options)

    def analyse(self, board: chess.Board, limit: chess.engine.Limit) -> dict:
        if self.fail:
            raise RuntimeError("stockfish failed")
        self.calls.append((board.fen(), limit.depth))
        if board.move_stack and board.peek().uci() == "e7e5":
            return {"score": chess.engine.PovScore(chess.engine.Cp(15), chess.BLACK)}
        if board.move_stack and board.peek().uci() == "d2d4":
            return {"score": chess.engine.PovScore(chess.engine.Cp(10), chess.WHITE)}
        if board.turn == chess.BLACK:
            return {
                "score": chess.engine.PovScore(chess.engine.Cp(20), chess.BLACK),
                "pv": [chess.Move.from_uci("e7e5")],
            }
        return {
            "score": chess.engine.PovScore(chess.engine.Cp(35), chess.WHITE),
            "pv": [chess.Move.from_uci("e2e4")],
        }

    def quit(self) -> None:
        self.quit_called = True


@pytest.fixture()
def client(tmp_path):
    settings = BackendSettings(
        artifact_root=tmp_path / "runs",
        book_root=tmp_path / "books",
        llm_base_url="http://example.test/v1",
        llm_api_key="test-key",
        llm_model="unit-model",
    )
    app = create_app(
        settings=settings,
        llm_client=StaticLLMClient("<think>Take the center.</think>\n<move>e7e5</move>"),
    )
    return TestClient(app)


@pytest.fixture()
def tool_client(tmp_path):
    settings = BackendSettings(
        artifact_root=tmp_path / "runs",
        book_root=tmp_path / "books",
        llm_base_url="http://example.test/v1",
        llm_api_key="test-key",
        llm_model="unit-model",
        stockfish_path=tmp_path / "stockfish.exe",
        stockfish_depth=18,
        stockfish_threads=2,
        stockfish_hash_mb=32,
        tool_batch_limit=2,
    )
    engine = FakeStockfish()
    app = create_app(
        settings=settings,
        llm_client=StaticLLMClient("<think>Mirror the center.</think>\n<move>e7e5</move>"),
        tool_service=ToolService(settings=settings, engine=engine),
    )
    return TestClient(app)


def test_health_and_books_endpoints(client, tmp_path):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert client.get("/api/books").status_code == 200


def test_tool_status_reports_stockfish_disabled_when_unconfigured(client):
    response = client.get("/api/tools")

    assert response.status_code == 200
    payload = response.json()
    assert payload["stockfish"]["enabled"] is False
    assert payload["stockfish"]["available"] is False
    assert payload["stockfish"]["depth"] == 16
    assert payload["capabilities"]["legal_moves"] is True
    assert payload["capabilities"]["opening_book_moves"] is True
    assert payload["capabilities"]["stockfish_analysis"] is False


def test_tool_status_reports_injected_stockfish_config(tool_client):
    response = tool_client.get("/api/tools")

    assert response.status_code == 200
    payload = response.json()
    assert payload["stockfish"]["enabled"] is True
    assert payload["stockfish"]["available"] is True
    assert payload["stockfish"]["name"] == "Fake Stockfish"
    assert payload["stockfish"]["depth"] == 18
    assert payload["stockfish"]["threads"] == 2
    assert payload["stockfish"]["hash_mb"] == 32
    assert payload["capabilities"]["stockfish_analysis"] is True
    assert tool_client.app.state.tool_service.engine.configured == [{"Threads": 2, "Hash": 32}]


def test_game_flow_rejects_illegal_human_move_and_persists_llm_artifact(client):
    created = client.post(
        "/api/games",
        json={"human_side": "white", "book_id": None, "book_max_plies": 0},
    )
    assert created.status_code == 200
    game_id = created.json()["game_id"]

    illegal = client.post(f"/api/games/{game_id}/human-move", json={"move_uci": "e2e5"})
    assert illegal.status_code == 400
    assert "illegal" in illegal.json()["detail"].lower()

    legal = client.post(f"/api/games/{game_id}/human-move", json={"move_uci": "e2e4"})
    assert legal.status_code == 200

    llm = client.post(f"/api/games/{game_id}/llm-move")
    assert llm.status_code == 200
    payload = llm.json()
    assert payload["last_rollout"]["parsed_answer"]["move_uci"] == "e7e5"
    assert payload["last_judgment"]["legal"] is True
    llm_move = payload["moves"][-1]
    assert llm_move["rollout"]["raw_output"] == "<think>Take the center.</think>\n<move>e7e5</move>"
    assert llm_move["rollout"]["parsed_answer"]["move_uci"] == "e7e5"
    assert llm_move["judgment"]["legal"] is True
    assert llm_move["judgment"]["rollout_id"] == llm_move["rollout_id"]

    runs = sorted((client.app.state.settings.artifact_root).glob("*/rollouts.jsonl"))
    assert len(runs) == 1
    assert "e7e5" in runs[0].read_text(encoding="utf-8")


def test_illegal_llm_move_sets_pending_recovery_and_does_not_advance_board(tmp_path):
    settings = BackendSettings(
        artifact_root=tmp_path / "runs",
        book_root=tmp_path / "books",
        llm_model="unit-model",
    )
    app = create_app(
        settings=settings,
        llm_client=StaticLLMClient("<think>Bad side to move.</think>\n<move>e7e5</move>"),
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
    assert payload["last_judgment"]["metadata"]["requires_recovery"] is True
    pending = payload["pending_recovery"]
    assert pending["rollout_id"] == payload["last_rollout"]["rollout_id"]
    assert pending["judgment_id"] == payload["last_judgment"]["judgment_id"]
    assert "illegal" in pending["reason"].lower()

    judgment_rows = sorted(settings.artifact_root.glob("*/judgments.jsonl"))
    assert len(judgment_rows) == 1
    assert '"requires_recovery": true' in judgment_rows[0].read_text(encoding="utf-8")


def test_recovery_retry_applies_legal_retry_and_records_metadata(tmp_path):
    llm = SequenceLLMClient(
        [
            "<think>Bad side to move.</think>\n<move>e7e5</move>",
            "<think>Develop a knight.</think>\n<move>g1f3</move>",
        ]
    )
    settings = BackendSettings(
        artifact_root=tmp_path / "runs",
        book_root=tmp_path / "books",
        llm_model="unit-model",
    )
    app = create_app(settings=settings, llm_client=llm)
    local_client = TestClient(app)
    created = local_client.post("/api/games", json={"human_side": "black"})
    game_id = created.json()["game_id"]
    failed = local_client.post(f"/api/games/{game_id}/llm-move").json()
    failed_rollout_id = failed["pending_recovery"]["rollout_id"]
    failed_judgment_id = failed["pending_recovery"]["judgment_id"]

    response = local_client.post(
        f"/api/games/{game_id}/recovery-move",
        json={"action": "retry"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pending_recovery"] is None
    assert payload["moves"][-1]["move_uci"] == "g1f3"
    assert payload["moves"][-1]["source"] == "llm"
    assert payload["last_judgment"]["legal"] is True
    assert payload["last_judgment"]["metadata"]["source"] == "ui_live_recovery"
    assert payload["last_judgment"]["metadata"]["recovery_action"] == "retry"
    assert payload["last_judgment"]["metadata"]["recovered_from_rollout_id"] == failed_rollout_id
    assert payload["last_judgment"]["metadata"]["recovered_from_judgment_id"] == failed_judgment_id


def test_recovery_retry_with_legal_moves_adds_legal_move_list_to_prompt(tmp_path):
    llm = SequenceLLMClient(
        [
            "<think>Bad side to move.</think>\n<move>e7e5</move>",
            "<think>Use the legal move list.</think>\n<move>b1c3</move>",
        ]
    )
    settings = BackendSettings(
        artifact_root=tmp_path / "runs",
        book_root=tmp_path / "books",
        llm_model="unit-model",
    )
    app = create_app(settings=settings, llm_client=llm)
    local_client = TestClient(app)
    created = local_client.post("/api/games", json={"human_side": "black"})
    game_id = created.json()["game_id"]
    assert local_client.post(f"/api/games/{game_id}/llm-move").status_code == 200

    response = local_client.post(
        f"/api/games/{game_id}/recovery-move",
        json={"action": "retry_with_legal_moves"},
    )

    assert response.status_code == 200
    assert response.json()["moves"][-1]["move_uci"] == "b1c3"
    retry_prompt = llm.requests[-1].messages[-1].content
    assert "Legal UCI moves:" in retry_prompt
    assert "b1c3" in retry_prompt
    assert "e7e5" not in retry_prompt


def test_recovery_legal_stub_applies_deterministic_move_after_failure(tmp_path):
    settings = BackendSettings(
        artifact_root=tmp_path / "runs",
        book_root=tmp_path / "books",
        llm_model="unit-model",
    )
    app = create_app(
        settings=settings,
        llm_client=StaticLLMClient("<think>Bad side to move.</think>\n<move>e7e5</move>"),
    )
    local_client = TestClient(app)
    created = local_client.post("/api/games", json={"human_side": "black"})
    game_id = created.json()["game_id"]
    assert local_client.post(f"/api/games/{game_id}/llm-move").status_code == 200

    response = local_client.post(
        f"/api/games/{game_id}/recovery-move",
        json={"action": "legal_stub"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pending_recovery"] is None
    assert payload["moves"][-1]["move_uci"] == "a2a3"
    assert payload["last_rollout"]["metadata"]["client"] == "deterministic_legal_stub"
    assert payload["last_judgment"]["metadata"]["recovery_action"] == "legal_stub"


def test_recovery_teacher_applies_stockfish_teacher_move_after_failure(tmp_path):
    settings = BackendSettings(
        artifact_root=tmp_path / "runs",
        book_root=tmp_path / "books",
        llm_model="unit-model",
        stockfish_path=tmp_path / "stockfish.exe",
    )
    engine = FakeStockfish()
    app = create_app(
        settings=settings,
        llm_client=StaticLLMClient("<think>Bad side to move.</think>\n<move>e7e5</move>"),
        tool_service=ToolService(settings=settings, engine=engine),
    )
    local_client = TestClient(app)
    created = local_client.post("/api/games", json={"human_side": "black"})
    game_id = created.json()["game_id"]
    failed = local_client.post(f"/api/games/{game_id}/llm-move").json()
    assert failed["last_judgment"]["teacher_move_uci"] == "e2e4"

    response = local_client.post(
        f"/api/games/{game_id}/recovery-move",
        json={"action": "teacher"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pending_recovery"] is None
    assert payload["moves"][-1]["move_uci"] == "e2e4"
    assert payload["last_judgment"]["metadata"]["recovery_action"] == "teacher"


def test_recovery_without_pending_failure_returns_400(client):
    created = client.post("/api/games", json={"human_side": "black"})
    game_id = created.json()["game_id"]

    response = client.post(
        f"/api/games/{game_id}/recovery-move",
        json={"action": "retry"},
    )

    assert response.status_code == 400
    assert "no pending" in response.json()["detail"].lower()


def test_live_llm_move_uses_stockfish_scoring_when_available(tool_client):
    created = tool_client.post(
        "/api/games",
        json={"human_side": "white", "book_id": None, "book_max_plies": 0},
    )
    game_id = created.json()["game_id"]
    assert tool_client.post(f"/api/games/{game_id}/human-move", json={"move_uci": "e2e4"}).status_code == 200

    response = tool_client.post(f"/api/games/{game_id}/llm-move")

    assert response.status_code == 200
    judgment = response.json()["last_judgment"]
    assert judgment["legal"] is True
    assert judgment["teacher_move_uci"] == "e7e5"
    assert judgment["regret_cp"] == 5.0
    assert judgment["metadata"]["judge"] == "stockfish"
    assert judgment["metadata"]["stockfish_depth"] == 18
    assert judgment["metadata"]["stockfish_scored"] is True

    judgments = sorted((tool_client.app.state.settings.artifact_root).glob("*/judgments.jsonl"))
    assert len(judgments) == 1
    persisted = judgments[0].read_text(encoding="utf-8")
    assert '"teacher_move_uci": "e7e5"' in persisted
    assert '"regret_cp": 5.0' in persisted


def test_live_llm_move_falls_back_to_legality_when_stockfish_unavailable(tmp_path):
    settings = BackendSettings(
        artifact_root=tmp_path / "runs",
        book_root=tmp_path / "books",
        llm_model="unit-model",
    )
    app = create_app(
        settings=settings,
        llm_client=StaticLLMClient("<think>Mirror.</think>\n<move>e7e5</move>"),
    )
    local_client = TestClient(app)
    created = local_client.post("/api/games", json={"human_side": "white"})
    game_id = created.json()["game_id"]
    assert local_client.post(f"/api/games/{game_id}/human-move", json={"move_uci": "e2e4"}).status_code == 200

    response = local_client.post(f"/api/games/{game_id}/llm-move")

    assert response.status_code == 200
    judgment = response.json()["last_judgment"]
    assert judgment["legal"] is True
    assert judgment["teacher_move_uci"] is None
    assert judgment["regret_cp"] is None
    assert judgment["metadata"]["judge"] == "legality"
    assert judgment["metadata"]["stockfish_available"] is False


def test_tools_analyze_returns_legal_moves_and_book_suggestions(client, tmp_path, monkeypatch):
    import chess_llm_ui.tools as tools_module

    book_root = client.app.state.settings.book_root
    book_root.mkdir(parents=True)
    (book_root / "Test.bin").write_bytes(b"placeholder")
    monkeypatch.setattr(
        tools_module,
        "get_weighted_book_moves",
        lambda path, board: [("e2e4", 12), ("d2d4", 6)],
    )

    response = client.post(
        "/api/tools/analyze",
        json={"fen": STARTING_FEN, "book_id": "Test", "include_stockfish": False},
    )

    assert response.status_code == 200
    payload = response.json()
    e4 = next(move for move in payload["legal_moves"] if move["uci"] == "e2e4")
    assert e4["san"] == "e4"
    assert e4["capture"] is False
    assert e4["check"] is False
    assert payload["legal_move_count"] == 20
    assert payload["book_moves"][0]["uci"] == "e2e4"
    assert payload["book_moves"][0]["san"] == "e4"
    assert payload["book_moves"][0]["weight"] == 12
    assert payload["stockfish"] is None


def test_artifact_loader_joins_prompt_rollout_and_judgment(client, tmp_path):
    artifact_dir = tmp_path / "artifact-run"
    prompt = PromptArtifact(
        prompt_id="prompt-1",
        fen=STARTING_FEN,
        task_type="best_move",
        messages=[ChatMessage(role="user", content="FEN: start")],
        metadata={"split": "planning"},
    )
    rollout = RolloutArtifact(
        rollout_id="rollout-1",
        prompt_id="prompt-1",
        model_id="unit-model",
        raw_output="<move>e2e4</move>",
        parsed_answer=ParsedAnswer(raw_text="<move>e2e4</move>", move_uci="e2e4", format_type="move_tag"),
    )
    judgment = JudgmentArtifact(
        judgment_id="judgment-1",
        rollout_id="rollout-1",
        legal=True,
        regret_cp=12.0,
        teacher_move_uci="e2e4",
        feedback="Good move.",
    )
    write_jsonl(artifact_dir / "prompts.jsonl", [prompt])
    write_jsonl(artifact_dir / "rollouts.jsonl", [rollout])
    write_jsonl(artifact_dir / "judgments.jsonl", [judgment])

    loaded = client.post("/api/artifacts/load", json={"artifact_dir": str(artifact_dir)})
    assert loaded.status_code == 200
    run_id = loaded.json()["run_id"]

    rows = client.get(f"/api/artifacts/{run_id}/rollouts").json()["items"]
    assert rows[0]["prompt"]["prompt_id"] == "prompt-1"
    assert rows[0]["rollout"]["rollout_id"] == "rollout-1"
    assert rows[0]["judgment"]["feedback"] == "Good move."

    detail = client.get(f"/api/artifacts/{run_id}/rollouts/rollout-1")
    assert detail.status_code == 200
    assert detail.json()["prompt"]["metadata"]["split"] == "planning"


def test_review_score_updates_selected_rollouts_in_memory(tool_client, tmp_path):
    artifact_dir = tmp_path / "artifact-run"
    prompt = PromptArtifact(
        prompt_id="prompt-1",
        fen=STARTING_FEN,
        task_type="best_move",
        messages=[ChatMessage(role="user", content="FEN: start")],
    )
    rollout = RolloutArtifact(
        rollout_id="rollout-1",
        prompt_id="prompt-1",
        model_id="unit-model",
        raw_output="<move>d2d4</move>",
        parsed_answer=ParsedAnswer(raw_text="<move>d2d4</move>", move_uci="d2d4", format_type="move_tag"),
    )
    judgment = JudgmentArtifact(
        judgment_id="judgment-old",
        rollout_id="rollout-1",
        legal=True,
        feedback="Old judgment.",
    )
    write_jsonl(artifact_dir / "prompts.jsonl", [prompt])
    write_jsonl(artifact_dir / "rollouts.jsonl", [rollout])
    write_jsonl(artifact_dir / "judgments.jsonl", [judgment])
    run_id = tool_client.post("/api/artifacts/load", json={"artifact_dir": str(artifact_dir)}).json()["run_id"]

    response = tool_client.post(
        f"/api/artifacts/{run_id}/score",
        json={"rollout_ids": ["rollout-1"], "depth": 18},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scored_count"] == 1
    assert payload["skipped_count"] == 0
    scored = payload["items"][0]["judgment"]
    assert scored["teacher_move_uci"] == "e2e4"
    assert scored["regret_cp"] == 25.0
    assert scored["metadata"]["judge"] == "stockfish"

    reloaded = tool_client.get(f"/api/artifacts/{run_id}/rollouts/rollout-1").json()
    assert reloaded["judgment"]["teacher_move_uci"] == "e2e4"


def test_review_score_rejects_batches_over_configured_limit(tool_client, tmp_path):
    artifact_dir = tmp_path / "artifact-run"
    _write_artifact_fixture(artifact_dir, "one")
    run_id = tool_client.post("/api/artifacts/load", json={"artifact_dir": str(artifact_dir)}).json()["run_id"]

    response = tool_client.post(
        f"/api/artifacts/{run_id}/score",
        json={"rollout_ids": ["rollout-one", "rollout-two", "rollout-three"]},
    )

    assert response.status_code == 400
    assert "batch limit" in response.json()["detail"].lower()


def test_artifact_loader_keeps_same_named_directories_distinct(client, tmp_path):
    first_dir = tmp_path / "first" / "artifact-run"
    second_dir = tmp_path / "second" / "artifact-run"
    _write_artifact_fixture(first_dir, "first")
    _write_artifact_fixture(second_dir, "second")

    first = client.post("/api/artifacts/load", json={"artifact_dir": str(first_dir)})
    second = client.post("/api/artifacts/load", json={"artifact_dir": str(second_dir)})

    assert first.status_code == 200
    assert second.status_code == 200
    first_run_id = first.json()["run_id"]
    second_run_id = second.json()["run_id"]
    assert first_run_id != second_run_id
    assert client.get(f"/api/artifacts/{first_run_id}/rollouts/rollout-first").status_code == 200
    assert client.get(f"/api/artifacts/{second_run_id}/rollouts/rollout-second").status_code == 200


def test_default_no_endpoint_uses_deterministic_legal_stub(tmp_path):
    settings = BackendSettings(
        artifact_root=tmp_path / "runs",
        book_root=tmp_path / "books",
        llm_model="unit-model",
    )
    app = create_app(settings=settings)
    local_client = TestClient(app)

    assert local_client.get("/api/health").json()["llm_mode"] == "legal_stub"
    created = local_client.post("/api/games", json={"human_side": "white"})
    game_id = created.json()["game_id"]
    assert local_client.post(f"/api/games/{game_id}/human-move", json={"move_uci": "e2e4"}).status_code == 200

    response = local_client.post(f"/api/games/{game_id}/llm-move")

    assert response.status_code == 200
    payload = response.json()
    assert payload["last_judgment"]["legal"] is True
    assert payload["last_rollout"]["parsed_answer"]["move_uci"] != "e2e4"
    assert payload["moves"][-1]["source"] == "llm"


def test_llm_client_failure_returns_clear_bad_gateway(tmp_path):
    settings = BackendSettings(
        artifact_root=tmp_path / "runs",
        book_root=tmp_path / "books",
        llm_model="unit-model",
    )
    app = create_app(settings=settings, llm_client=FailingLLMClient())
    local_client = TestClient(app, raise_server_exceptions=False)
    created = local_client.post("/api/games", json={"human_side": "black"})
    game_id = created.json()["game_id"]

    response = local_client.post(f"/api/games/{game_id}/llm-move")

    assert response.status_code == 502
    assert "endpoint down" in response.json()["detail"]


def test_artifact_loader_rejects_malformed_jsonl_with_clear_400(client, tmp_path):
    artifact_dir = tmp_path / "bad-artifact-run"
    artifact_dir.mkdir()
    (artifact_dir / "prompts.jsonl").write_text("{not-json}\n", encoding="utf-8")
    (artifact_dir / "rollouts.jsonl").write_text("", encoding="utf-8")
    (artifact_dir / "judgments.jsonl").write_text("", encoding="utf-8")

    response = client.post("/api/artifacts/load", json={"artifact_dir": str(artifact_dir)})

    assert response.status_code == 400
    assert "invalid artifact" in response.json()["detail"].lower()


def test_concurrent_llm_requests_for_same_game_are_serialized(tmp_path):
    llm_client = BlockingLegalLLMClient()
    settings = BackendSettings(
        artifact_root=tmp_path / "runs",
        book_root=tmp_path / "books",
        llm_model="unit-model",
    )
    app = create_app(settings=settings, llm_client=llm_client)
    local_client = TestClient(app, raise_server_exceptions=False)
    created = local_client.post("/api/games", json={"human_side": "white"})
    game_id = created.json()["game_id"]
    assert local_client.post(f"/api/games/{game_id}/human-move", json={"move_uci": "e2e4"}).status_code == 200

    def request_llm_move() -> int:
        return local_client.post(f"/api/games/{game_id}/llm-move").status_code

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(request_llm_move)
        assert llm_client.entered.wait(timeout=2.0)
        second = executor.submit(request_llm_move)
        deadline = time.monotonic() + 1.0
        while llm_client.calls == 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert llm_client.calls == 1
        llm_client.release.set()
        statuses = sorted([first.result(timeout=3.0), second.result(timeout=3.0)])

    game = local_client.get(f"/api/games/{game_id}").json()
    assert statuses == [200, 400]
    assert llm_client.calls == 1
    assert [move["source"] for move in game["moves"]].count("llm") == 1


def _write_artifact_fixture(artifact_dir: Path, suffix: str) -> None:
    prompt = PromptArtifact(
        prompt_id=f"prompt-{suffix}",
        fen=STARTING_FEN,
        task_type="best_move",
        messages=[ChatMessage(role="user", content="FEN: start")],
    )
    rollout = RolloutArtifact(
        rollout_id=f"rollout-{suffix}",
        prompt_id=prompt.prompt_id,
        model_id="unit-model",
        raw_output="<move>e2e4</move>",
        parsed_answer=ParsedAnswer(raw_text="<move>e2e4</move>", move_uci="e2e4", format_type="move_tag"),
    )
    judgment = JudgmentArtifact(
        judgment_id=f"judgment-{suffix}",
        rollout_id=rollout.rollout_id,
        legal=True,
    )
    write_jsonl(artifact_dir / "prompts.jsonl", [prompt])
    write_jsonl(artifact_dir / "rollouts.jsonl", [rollout])
    write_jsonl(artifact_dir / "judgments.jsonl", [judgment])
