import json
from pathlib import Path
from types import SimpleNamespace

import chess
import chess.engine

from chess_llm.artifacts.jsonl import read_jsonl
from chess_llm.artifacts.schemas import JudgmentArtifact, PromptArtifact, RolloutArtifact
from chess_llm.autodata.failure_buckets import ILLEGAL_MOVE, PARSE_FAILURE
from chess_llm.autodata.selfplay import SelfPlayConfig, main, run_self_play
from chess_llm.autodata.sft_refresh import build_sft_refresh
from chess_llm.sft.fen_pool import FENPool
from chess_llm.sft.sources.self_play import POSITION_RECORD_KEYS, load_self_play_positions


class ScriptedMoveGenerator:
    """Canned outputs per game; defaults to the first sorted legal move."""

    def __init__(self, outputs_by_game: dict[int, list[str]] | None = None) -> None:
        self.outputs_by_game = {
            game_index: list(outputs)
            for game_index, outputs in (outputs_by_game or {}).items()
        }
        self.prompt_ids: list[str] = []

    def generate(self, prompts, *, temperature, max_new_tokens, seed):
        outputs = []
        for prompt in prompts:
            self.prompt_ids.append(prompt.prompt_id)
            queue = self.outputs_by_game.get(prompt.metadata["game_index"])
            if queue:
                outputs.append(queue.pop(0))
                continue
            board = chess.Board(prompt.fen)
            move_uci = sorted(move.uci() for move in board.legal_moves)[0]
            outputs.append(f"<think>scripted</think>\n<move>{move_uci}</move>")
        return outputs


class FakeJudgeEngine:
    """Analyse fake mirroring tests/test_stockfish_judge.py's FakeStockfish."""

    def __init__(self, best_cp: int = 200, move_cp: int = 50) -> None:
        self.best_cp = best_cp
        self.move_cp = move_cp
        self.calls: list[tuple[str, int | None]] = []

    def analyse(self, board: chess.Board, limit: chess.engine.Limit) -> dict:
        self.calls.append((board.fen(), limit.depth))
        if len(board.move_stack) == 0:
            return {
                "score": chess.engine.PovScore(chess.engine.Cp(self.best_cp), chess.WHITE),
                "pv": [next(iter(board.legal_moves))],
            }
        return {
            "score": chess.engine.PovScore(chess.engine.Cp(self.move_cp), chess.WHITE)
        }


class FakeOpponentEngine:
    """Opponent engine fake whose play() returns the first sorted legal move."""

    def __init__(self) -> None:
        self.configured: list[dict] = []
        self.play_limits: list[chess.engine.Limit] = []
        self.options = {"UCI_Elo": SimpleNamespace(min=1320, max=3190)}
        self.quit_called = False

    def configure(self, options: dict) -> None:
        self.configured.append(dict(options))

    def play(self, board: chess.Board, limit: chess.engine.Limit) -> SimpleNamespace:
        self.play_limits.append(limit)
        move_uci = sorted(move.uci() for move in board.legal_moves)[0]
        return SimpleNamespace(move=chess.Move.from_uci(move_uci))

    def quit(self) -> None:
        self.quit_called = True


def _read_dict_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _config(tmp_path: Path, **overrides) -> SelfPlayConfig:
    defaults = dict(
        model_id="test-model",
        games=2,
        batch_games=2,
        opponent="self",
        judge_depth=12,
        max_plies=10,
        output_dir=tmp_path / "self_play",
        run_id="testrun",
        seed=7,
        backend="legal_stub",
    )
    defaults.update(overrides)
    return SelfPlayConfig(**defaults)


def test_two_game_run_writes_triple_games_and_positions(tmp_path: Path):
    result = run_self_play(
        _config(tmp_path),
        generator=ScriptedMoveGenerator(),
        judge_engine=FakeJudgeEngine(),
    )

    prompts = list(read_jsonl(result.prompts_path, PromptArtifact))
    rollouts = list(read_jsonl(result.rollouts_path, RolloutArtifact))
    judgments = list(read_jsonl(result.judgments_path, JudgmentArtifact))
    games = _read_dict_jsonl(result.games_path)
    positions = _read_dict_jsonl(result.positions_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.game_count == 2
    assert len(games) == 2
    assert len(prompts) == len(rollouts) == len(judgments)
    assert len(prompts) == result.prompt_count > 0
    assert manifest["artifact_type"] == "self_play_manifest"
    assert manifest["position_count"] == len(positions)

    prompt_ids = [prompt.prompt_id for prompt in prompts]
    assert len(prompt_ids) == len(set(prompt_ids))
    assert all(prompt_id.startswith("selfplay-testrun-g") for prompt_id in prompt_ids)
    for prompt in prompts:
        assert prompt.task_type == "best_move"
        user_messages = [m for m in prompt.messages if m.role == "user"]
        assert user_messages
        assert user_messages[0].content == (
            f"FEN: {prompt.fen}\nWhat is the best move?"
        )

    # Every rollout corresponds to a model ply that actually landed a move.
    model_positions = [pos for pos in positions if pos["mover"] == "model"]
    assert len(rollouts) == len(model_positions)

    # Positions replay legally from the start FEN of each game.
    for game in games:
        game_positions = sorted(
            (pos for pos in positions if pos["game_id"] == game["game_id"]),
            key=lambda pos: pos["ply"],
        )
        assert len(game_positions) == game["n_plies"] == len(game["moves"])
        board = chess.Board(game["start_fen"])
        for position, move_uci in zip(game_positions, game["moves"]):
            assert position["fen"] == board.fen()
            assert position["move_played_uci"] == move_uci
            move = chess.Move.from_uci(move_uci)
            assert move in board.legal_moves
            board.push(move)

    # game_id is consistent across games.jsonl, positions, and artifacts.
    game_ids = {game["game_id"] for game in games}
    assert len(game_ids) == 2
    assert {pos["game_id"] for pos in positions} == game_ids
    for artifact in (*prompts, *rollouts, *judgments):
        assert artifact.metadata["game_id"] in game_ids


def test_illegal_outputs_backfill_teacher_and_hit_failure_cap(tmp_path: Path):
    judge = FakeJudgeEngine()
    result = run_self_play(
        _config(
            tmp_path,
            games=1,
            batch_games=1,
            max_model_failures=2,
            max_plies=30,
        ),
        generator=ScriptedMoveGenerator(
            {0: ["there is no chess content here", "<move>a1a1</move>"]}
        ),
        judge_engine=judge,
    )

    judgments = list(read_jsonl(result.judgments_path, JudgmentArtifact))
    games = _read_dict_jsonl(result.games_path)
    positions = _read_dict_jsonl(result.positions_path)

    assert [judgment.failure_bucket for judgment in judgments] == [
        PARSE_FAILURE,
        ILLEGAL_MOVE,
    ]
    for judgment in judgments:
        assert judgment.teacher_move_uci is not None
        assert judgment.metadata["teacher_move_source"] == "driver_backfill"

    assert games[0]["termination"] == "failure_cap"
    assert games[0]["result"] == "*"
    fallback_positions = [
        pos for pos in positions if pos["mover"] == "random_fallback"
    ]
    assert len(fallback_positions) == 2
    assert not [pos for pos in positions if pos["mover"] == "model"]


def test_adjudication_terminates_after_consecutive_judged_plies(tmp_path: Path):
    result = run_self_play(
        _config(
            tmp_path,
            games=1,
            batch_games=1,
            max_plies=60,
            adjudicate_cp=1000.0,
            adjudicate_plies=3,
        ),
        generator=ScriptedMoveGenerator(),
        judge_engine=FakeJudgeEngine(best_cp=1500, move_cp=1500),
    )

    games = _read_dict_jsonl(result.games_path)
    assert games[0]["termination"] == "adjudicated"
    assert games[0]["result"] == "1-0"
    # Exactly adjudicate_plies model plies were judged before termination.
    assert result.rollout_count == 3
    book_plies = games[0]["n_plies"] - 3
    assert 2 <= book_plies <= 6


def test_same_seed_produces_identical_move_sequences(tmp_path: Path):
    runs = []
    for name in ("first", "second"):
        result = run_self_play(
            _config(
                tmp_path,
                games=3,
                batch_games=2,
                judge_depth=0,
                output_dir=tmp_path / name,
            ),
            generator=None,
        )
        runs.append(_read_dict_jsonl(result.games_path))

    first, second = runs
    assert [game["moves"] for game in first] == [game["moves"] for game in second]
    assert [game["game_id"] for game in first] == [game["game_id"] for game in second]


def test_stockfish_opponent_plays_with_clamped_elo(tmp_path: Path):
    opponent = FakeOpponentEngine()
    result = run_self_play(
        _config(
            tmp_path,
            games=2,
            batch_games=2,
            opponent="stockfish",
            judge_depth=0,
            max_plies=8,
            opponent_elo_min=1000,
            opponent_elo_max=1500,
            opponent_nodes=777,
        ),
        generator=ScriptedMoveGenerator(),
        opponent_engine=opponent,
    )

    games = _read_dict_jsonl(result.games_path)
    positions = _read_dict_jsonl(result.positions_path)

    assert [game["model_side"] for game in games] == ["white", "black"]
    for game in games:
        assert game["opponent"] == "stockfish"
        # Sampled from [1000, 1500] but clamped to the engine minimum of 1320.
        assert 1320 <= game["opponent_elo"] <= 1500
    assert {pos["mover"] for pos in positions} >= {"model", "stockfish"}
    assert opponent.play_limits
    assert all(limit.nodes == 777 for limit in opponent.play_limits)
    elo_configs = [
        options for options in opponent.configured if "UCI_Elo" in options
    ]
    assert elo_configs
    assert all(options["UCI_LimitStrength"] is True for options in elo_configs)
    assert {options["UCI_Elo"] for options in elo_configs} == {
        game["opponent_elo"] for game in games
    }
    # Injected engines are not closed by the driver.
    assert opponent.quit_called is False


def test_selfplay_triple_feeds_build_sft_refresh(tmp_path: Path):
    result = run_self_play(
        _config(
            tmp_path,
            games=1,
            batch_games=1,
            max_plies=8,
        ),
        generator=ScriptedMoveGenerator(
            {0: ["no move to see here", "<move>a1a1</move>"]}
        ),
        judge_engine=FakeJudgeEngine(best_cp=200, move_cp=50),
    )

    blocklist_path = tmp_path / "blocklist.txt"
    blocklist_path.write_text("", encoding="utf-8")
    refresh = build_sft_refresh(
        result.prompts_path,
        result.rollouts_path,
        result.judgments_path,
        tmp_path / "refresh",
        blocklist_path=blocklist_path,
    )

    assert refresh.format_repair_count >= 1
    assert refresh.move_correction_count >= 1
    format_rows = _read_dict_jsonl(refresh.format_repair_path)
    assert format_rows[0]["task"] == "7.4_autodata_format_repair"
    assert format_rows[0]["metadata"]["source_prompt_id"].startswith(
        "selfplay-testrun-"
    )


def test_cli_smoke_legal_stub_run_supports_refresh_and_fen_pool(tmp_path: Path):
    output_dir = tmp_path / "self_play"
    assert (
        main(
            [
                "--backend",
                "legal_stub",
                "--opponent",
                "self",
                "--games",
                "4",
                "--max-plies",
                "12",
                "--judge-depth",
                "0",
                "--seed",
                "7",
                "--model-id",
                "smoke",
                "--output",
                str(output_dir),
                "--run-id",
                "smoke1",
            ]
        )
        == 0
    )

    run_dir = output_dir / "smoke1"
    blocklist_path = tmp_path / "blocklist.txt"
    blocklist_path.write_text("", encoding="utf-8")
    refresh = build_sft_refresh(
        run_dir / "prompts.jsonl",
        run_dir / "rollouts.jsonl",
        run_dir / "judgments.jsonl",
        tmp_path / "refresh",
        blocklist_path=blocklist_path,
    )
    assert refresh.skipped_count >= 0

    rows = load_self_play_positions(output_dir)
    assert rows
    pool = FENPool()
    for row in rows:
        assert all(key in row for key in POSITION_RECORD_KEYS)
        pool.add(
            row["fen"],
            source="self_play",
            game_phase=row["game_phase"],
            game_id=row["game_id"],
        )
    assert len(pool) > 0
