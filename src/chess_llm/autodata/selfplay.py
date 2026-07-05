"""Headless self-play driver that harvests judged positions for SFT refresh.

Runs N parallel games between the model and a strength-limited Stockfish
opponent (or the model itself), batching one generation call per ply round.
Every model ply is recorded as prompt/rollout/judgment artifacts compatible
with :mod:`chess_llm.autodata.sft_refresh`, and every played move becomes a
position record compatible with the SFT FEN pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any, Protocol, Sequence

import chess
import chess.engine

from chess_llm.artifacts.jsonl import write_jsonl
from chess_llm.artifacts.schemas import (
    ChatMessage,
    JudgmentArtifact,
    PromptArtifact,
    RolloutArtifact,
)
from chess_llm.autodata.failure_buckets import ILLEGAL_MOVE, PARSE_FAILURE
from chess_llm.autodata.judge import judge_rollout
from chess_llm.autodata.rollouts import build_rollout
from chess_llm.autodata.stockfish_judge import judge_rollout_with_stockfish
from chess_llm.core.opening_books import (
    OpeningBook,
    discover_opening_books,
    sample_polyglot_book_line,
)
from chess_llm.evals.benchmark import CANONICAL_PROMPTS, BenchmarkExample
from chess_llm.external.stockfish import (
    StockfishEngineConfig,
    clamp_uci_elo,
    configure_stockfish_engine,
    open_stockfish,
    stockfish_engine_name,
)
from chess_llm.formats.prompts import SYSTEM_PROMPT
from chess_llm.inference import (
    DeterministicLegalMoveClient,
    InferenceClient,
    InferenceRequest,
)
from chess_llm.sft.settings import SftDataSettings
from chess_llm.sft.sources.lichess_games import game_phase, material_balance

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SETTINGS_ROOT = _REPO_ROOT
DEFAULT_SELF_PLAY_OUTPUT_DIR = SftDataSettings.from_env(_SETTINGS_ROOT).self_play_dir

BEST_MOVE_PROMPT_TEMPLATE = CANONICAL_PROMPTS["best_move"]
SELF_PLAY_TASK_TYPE = "best_move"
OPPONENT_CHOICES = ("stockfish", "self", "mixed")
BACKEND_CHOICES = ("vllm", "transformers", "legal_stub")
MOVER_MODEL = "model"
MOVER_STOCKFISH = "stockfish"
MOVER_BOOK = "book"
MOVER_RANDOM_FALLBACK = "random_fallback"


class BatchMoveGenerator(Protocol):
    """Backend-agnostic batched move generation for self-play prompts."""

    def generate(
        self,
        prompts: Sequence[PromptArtifact],
        *,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[str]:
        """Return one raw model output per prompt, in prompt order."""


@dataclass(frozen=True)
class SelfPlayConfig:
    """Runtime configuration for one self-play harvesting run."""

    model_id: str
    model: str | None = None
    games: int = 128
    batch_games: int = 64
    opponent: str = "mixed"
    self_play_fraction: float = 0.25
    stockfish_path: str | None = None
    opponent_elo_min: int = 1320
    opponent_elo_max: int = 2000
    opponent_nodes: int = 10_000
    judge_depth: int = 12
    max_plies: int = 200
    max_model_failures: int = 20
    adjudicate_cp: float = 1000.0
    adjudicate_plies: int = 6
    book_dir: str | None = None
    book_plies: int = 8
    output_dir: str | Path | None = None
    run_id: str | None = None
    seed: int = 42
    temperature: float = 0.7
    max_new_tokens: int = 512
    backend: str = "vllm"


@dataclass
class GameState:
    """Mutable state for one in-flight self-play game."""

    game_index: int
    rng: Random
    opponent: str
    model_side: bool
    opponent_elo: int | None = None
    board: chess.Board = field(default_factory=chess.Board)
    start_fen: str = chess.STARTING_FEN
    moves: list[str] = field(default_factory=list)
    positions: list[dict] = field(default_factory=list)
    prompts: list[PromptArtifact] = field(default_factory=list)
    rollouts: list[RolloutArtifact] = field(default_factory=list)
    judgments: list[JudgmentArtifact] = field(default_factory=list)
    model_failures: int = 0
    adjudication_streak: int = 0
    adjudication_sign: int = 0
    result: str | None = None
    termination: str | None = None
    game_id: str | None = None

    @property
    def done(self) -> bool:
        return self.termination is not None


@dataclass(frozen=True)
class SelfPlayResult:
    """Paths and counts produced by one self-play run."""

    run_dir: Path
    prompts_path: Path
    rollouts_path: Path
    judgments_path: Path
    games_path: Path
    positions_path: Path
    manifest_path: Path
    game_count: int
    prompt_count: int
    rollout_count: int
    judgment_count: int
    position_count: int
    terminations: dict[str, int]


class ClientMoveGenerator:
    """Sequential adapter over a :class:`chess_llm.inference.InferenceClient`."""

    def __init__(self, client: InferenceClient, *, model_id: str) -> None:
        self.client = client
        self.model_id = model_id

    def generate(
        self,
        prompts: Sequence[PromptArtifact],
        *,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[str]:
        outputs: list[str] = []
        for prompt in prompts:
            request = InferenceRequest(
                model_id=self.model_id,
                messages=list(prompt.messages),
                temperature=temperature,
                max_tokens=max_new_tokens,
                metadata={"fen": prompt.fen, "seed": seed},
            )
            outputs.append(self.client.complete(request).raw_text)
        return outputs


class VllmMoveGenerator:
    """Batched self-play generation over vLLM offline inference."""

    def __init__(
        self,
        model_path: str,
        *,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
    ) -> None:
        # Heavy import kept lazy: pulls in vLLM/torch/transformers.
        from chess_llm.training.evaluate import (
            VllmPredictionGenerator,
            load_prompt_tokenizer,
        )

        tokenizer = load_prompt_tokenizer(model_path)
        self._generator = VllmPredictionGenerator(
            model_path,
            tokenizer,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

    def generate(
        self,
        prompts: Sequence[PromptArtifact],
        *,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[str]:
        examples = [_benchmark_example(prompt) for prompt in prompts]
        predictions = self._generator.generate(
            examples,
            num_samples=1,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        return [predictions[example.example_id][0] for example in examples]


class TransformersMoveGenerator:
    """Batched self-play generation over the HF transformers fallback."""

    def __init__(
        self,
        model_path: str,
        *,
        batch_size: int = 16,
        attn_implementation: str = "auto",
    ) -> None:
        # Heavy import kept lazy: pulls in torch/transformers.
        from chess_llm.training import evaluate as training_evaluate

        self._evaluate = training_evaluate
        self.batch_size = batch_size
        self.model, self.tokenizer = training_evaluate.load_model_and_tokenizer(
            model_path,
            attn_implementation=attn_implementation,
        )

    def generate(
        self,
        prompts: Sequence[PromptArtifact],
        *,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[str]:
        examples = [_benchmark_example(prompt) for prompt in prompts]
        predictions = self._evaluate.generate_predictions_transformers(
            self.model,
            self.tokenizer,
            examples,
            num_samples=1,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            batch_size=self.batch_size,
            seed=seed,
        )
        return [predictions[example.example_id][0] for example in examples]


def build_move_generator(config: SelfPlayConfig) -> BatchMoveGenerator:
    """Construct the batched generation backend for this run."""
    if config.backend == "legal_stub":
        return ClientMoveGenerator(
            DeterministicLegalMoveClient(model_id=config.model_id),
            model_id=config.model_id,
        )
    if not config.model:
        raise ValueError(f"--model is required for backend {config.backend!r}")
    if config.backend == "vllm":
        return VllmMoveGenerator(config.model)
    if config.backend == "transformers":
        return TransformersMoveGenerator(config.model)
    raise ValueError(f"unsupported backend: {config.backend!r}")


def run_self_play(
    config: SelfPlayConfig,
    *,
    generator: BatchMoveGenerator | None = None,
    opponent_engine: Any | None = None,
    judge_engine: Any | None = None,
) -> SelfPlayResult:
    """Play ``config.games`` games and write the artifact/position files."""
    if config.output_dir is None:
        raise ValueError("output_dir is required")
    if config.opponent not in OPPONENT_CHOICES:
        raise ValueError(f"unsupported opponent mode: {config.opponent!r}")
    if config.run_id is None:
        config = replace(
            config,
            run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        )

    master_rng = Random(config.seed)
    opponents = [_assign_opponent(config, master_rng) for _ in range(config.games)]

    if generator is None:
        generator = build_move_generator(config)
    books = discover_opening_books(config.book_dir) if config.book_dir else []

    use_judge = config.judge_depth > 0
    needs_opponent = any(opponent == "stockfish" for opponent in opponents)
    opened_opponent = False
    opened_judge = False
    games: list[GameState] = []
    try:
        if needs_opponent and opponent_engine is None:
            opponent_engine = _open_engine(config, role="opponent")
            opened_opponent = True
        if use_judge and judge_engine is None:
            # A separate full-strength handle: UCI_LimitStrength on the
            # opponent engine would also weaken judging analyse calls.
            judge_engine = _open_engine(config, role="judge")
            opened_judge = True
        active_judge = judge_engine if use_judge else None
        judge_engine_name = (
            stockfish_engine_name(active_judge) if active_judge is not None else None
        )

        batch_size = max(1, config.batch_games)
        for batch_start in range(0, config.games, batch_size):
            batch = [
                _new_game(index, opponents[index], config, opponent_engine)
                for index in range(
                    batch_start, min(batch_start + batch_size, config.games)
                )
            ]
            for game in batch:
                _apply_opening(game, config, books)
            _run_batch(batch, generator, opponent_engine, active_judge, config)
            for game in batch:
                _finalize_game(game)
            games.extend(batch)
    finally:
        if opened_opponent and opponent_engine is not None:
            opponent_engine.quit()
        if opened_judge and judge_engine is not None:
            judge_engine.quit()

    return _write_outputs(
        games,
        config,
        judge_mode="stockfish" if use_judge else "legality",
        judge_engine_name=judge_engine_name,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``python -m chess_llm.autodata.selfplay``."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    config = SelfPlayConfig(
        model_id=args.model_id,
        model=args.model,
        games=args.games,
        batch_games=args.batch_games,
        opponent=args.opponent,
        self_play_fraction=args.self_play_fraction,
        stockfish_path=args.stockfish_path,
        opponent_elo_min=args.opponent_elo_min,
        opponent_elo_max=args.opponent_elo_max,
        opponent_nodes=args.opponent_nodes,
        judge_depth=args.judge_depth,
        max_plies=args.max_plies,
        max_model_failures=args.max_model_failures,
        adjudicate_cp=args.adjudicate_cp,
        adjudicate_plies=args.adjudicate_plies,
        book_dir=args.book_dir,
        book_plies=args.book_plies,
        output_dir=args.output,
        run_id=args.run_id,
        seed=args.seed,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        backend=args.backend,
    )
    result = run_self_play(config)
    logger.info(
        "Self-play run complete: %d games, %d model plies, %d positions -> %s",
        result.game_count,
        result.rollout_count,
        result.position_count,
        result.run_dir,
    )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Harvest judged self-play positions for SFT refresh."
    )
    parser.add_argument("--model", default=None, help="Model checkpoint path for GPU backends.")
    parser.add_argument("--model-id", required=True, help="Model identifier stored on artifacts.")
    parser.add_argument("--games", type=int, default=128, help="Number of games to play.")
    parser.add_argument(
        "--batch-games",
        type=int,
        default=64,
        help="Games kept in flight per batched generation round.",
    )
    parser.add_argument(
        "--opponent",
        choices=OPPONENT_CHOICES,
        default="mixed",
        help="Opponent assignment mode.",
    )
    parser.add_argument(
        "--self-play-fraction",
        type=float,
        default=0.25,
        help="Fraction of model-vs-model games in --opponent mixed mode.",
    )
    parser.add_argument(
        "--stockfish-path",
        default=None,
        help="Stockfish binary for the opponent and judge engines.",
    )
    parser.add_argument("--opponent-elo-min", type=int, default=1320)
    parser.add_argument("--opponent-elo-max", type=int, default=2000)
    parser.add_argument(
        "--opponent-nodes",
        type=int,
        default=10_000,
        help="Node limit per opponent engine move.",
    )
    parser.add_argument(
        "--judge-depth",
        type=int,
        default=12,
        help="Stockfish judging depth; 0 judges legality only.",
    )
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--max-model-failures", type=int, default=20)
    parser.add_argument(
        "--adjudicate-cp",
        type=float,
        default=1000.0,
        help="White-centric centipawn threshold for adjudication.",
    )
    parser.add_argument(
        "--adjudicate-plies",
        type=int,
        default=6,
        help="Consecutive judged plies past the threshold before adjudication.",
    )
    parser.add_argument("--book-dir", default=None, help="Polyglot .bin book directory.")
    parser.add_argument(
        "--book-plies",
        type=int,
        default=8,
        help="Maximum opening book plies sampled per game.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_SELF_PLAY_OUTPUT_DIR),
        help="Root directory for per-run output folders.",
    )
    parser.add_argument("--run-id", default=None, help="Run folder name (default UTC timestamp).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default="vllm",
        help="Generation backend; legal_stub needs no GPU or checkpoint.",
    )
    return parser


def _assign_opponent(config: SelfPlayConfig, master_rng: Random) -> str:
    if config.opponent == "mixed":
        return "self" if master_rng.random() < config.self_play_fraction else "stockfish"
    return config.opponent


def _open_engine(config: SelfPlayConfig, *, role: str) -> Any:
    if not config.stockfish_path:
        raise ValueError(f"--stockfish-path is required for the {role} engine")
    return open_stockfish(StockfishEngineConfig(path=config.stockfish_path))


def _new_game(
    game_index: int,
    opponent: str,
    config: SelfPlayConfig,
    opponent_engine: Any | None,
) -> GameState:
    game_rng = Random(config.seed * 10_000 + game_index)
    opponent_elo: int | None = None
    if opponent == "stockfish":
        elo_low = min(config.opponent_elo_min, config.opponent_elo_max)
        elo_high = max(config.opponent_elo_min, config.opponent_elo_max)
        opponent_elo = game_rng.randint(elo_low, elo_high)
        if opponent_engine is not None:
            opponent_elo = clamp_uci_elo(opponent_engine, opponent_elo)
    return GameState(
        game_index=game_index,
        rng=game_rng,
        opponent=opponent,
        model_side=chess.WHITE if game_index % 2 == 0 else chess.BLACK,
        opponent_elo=opponent_elo,
    )


def _apply_opening(
    game: GameState,
    config: SelfPlayConfig,
    books: Sequence[OpeningBook],
) -> None:
    if books:
        plies = game.rng.randint(0, max(0, config.book_plies))
        if plies <= 0:
            return
        book = game.rng.choice(list(books))
        line = sample_polyglot_book_line(
            book.path,
            max_plies=plies,
            seed=game.rng.randrange(2**31),
        )
        for sampled in line.moves:
            if game.done:
                return
            _push_move(game, chess.Move.from_uci(sampled.move_uci), MOVER_BOOK, config)
        return

    for _ in range(game.rng.randint(2, 6)):
        if game.done:
            return
        move_uci = _random_legal_move(game)
        if move_uci is None:
            return
        _push_move(game, chess.Move.from_uci(move_uci), MOVER_BOOK, config)


def _run_batch(
    batch: list[GameState],
    generator: BatchMoveGenerator,
    opponent_engine: Any | None,
    judge_engine: Any | None,
    config: SelfPlayConfig,
) -> None:
    round_index = 0
    while True:
        for game in batch:
            while not game.done and not _is_model_turn(game):
                _play_opponent_move(game, opponent_engine, config)
        pending = [game for game in batch if not game.done]
        if not pending:
            return
        prompts = [_build_prompt(game, config) for game in pending]
        outputs = generator.generate(
            prompts,
            temperature=config.temperature,
            max_new_tokens=config.max_new_tokens,
            seed=config.seed + round_index,
        )
        if len(outputs) != len(prompts):
            raise ValueError(
                f"generator returned {len(outputs)} outputs for {len(prompts)} prompts"
            )
        for game, prompt, raw_output in zip(pending, prompts, outputs):
            _apply_model_output(game, prompt, raw_output, judge_engine, config)
        round_index += 1


def _is_model_turn(game: GameState) -> bool:
    return game.opponent == "self" or game.board.turn == game.model_side


def _play_opponent_move(
    game: GameState,
    opponent_engine: Any | None,
    config: SelfPlayConfig,
) -> None:
    if opponent_engine is None:
        raise ValueError("stockfish opponent games require an opponent engine")
    # Per-move reconfiguration: interleaved games can carry different Elos.
    configure_stockfish_engine(opponent_engine, limit_strength_elo=game.opponent_elo)
    play_result = opponent_engine.play(
        game.board,
        chess.engine.Limit(nodes=config.opponent_nodes),
    )
    _push_move(game, play_result.move, MOVER_STOCKFISH, config)


def _build_prompt(game: GameState, config: SelfPlayConfig) -> PromptArtifact:
    ply = len(game.moves)
    fen = game.board.fen()
    return PromptArtifact(
        prompt_id=_artifact_id(config, game, ply),
        messages=[
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=BEST_MOVE_PROMPT_TEMPLATE.format(fen=fen)),
        ],
        fen=fen,
        task_type=SELF_PLAY_TASK_TYPE,
        metadata={
            "split": "selfplay",
            "run_id": config.run_id,
            "game_index": game.game_index,
            "ply": ply,
            "opponent": game.opponent,
            "model_side": "white" if game.model_side == chess.WHITE else "black",
        },
    )


def _apply_model_output(
    game: GameState,
    prompt: PromptArtifact,
    raw_output: str,
    judge_engine: Any | None,
    config: SelfPlayConfig,
) -> None:
    artifact_metadata = {
        "run_id": config.run_id,
        "game_index": game.game_index,
        "ply": prompt.metadata["ply"],
    }
    rollout = build_rollout(
        prompt,
        config.model_id,
        raw_output,
        rollout_id=prompt.prompt_id,
        metadata=dict(artifact_metadata),
    )
    if judge_engine is not None:
        judgment = judge_rollout_with_stockfish(
            prompt,
            rollout,
            judge_engine,
            judgment_id=prompt.prompt_id,
            depth=config.judge_depth,
            metadata=dict(artifact_metadata),
        )
        if judgment.teacher_move_uci is None and judgment.failure_bucket in (
            PARSE_FAILURE,
            ILLEGAL_MOVE,
        ):
            # judge_rollout_with_stockfish never analyses failed rollouts, but
            # sft_refresh drops correction rows without a teacher target, so
            # backfill one analyse of the pre-move board here.
            judgment = _backfill_teacher_move(prompt, judgment, judge_engine, config)
    else:
        judgment = judge_rollout(
            prompt,
            rollout,
            judgment_id=prompt.prompt_id,
            metadata=dict(artifact_metadata),
        )
    game.prompts.append(prompt)
    game.rollouts.append(rollout)
    game.judgments.append(judgment)

    move = _legal_parsed_move(game.board, rollout.parsed_answer.move_uci)
    if move is not None:
        mover_color = game.board.turn
        _push_move(game, move, MOVER_MODEL, config)
        if not game.done:
            _update_adjudication(game, judgment, mover_color, config)
        return

    game.model_failures += 1
    game.adjudication_streak = 0
    fallback_uci = _random_legal_move(game)
    if fallback_uci is not None:
        _push_move(game, chess.Move.from_uci(fallback_uci), MOVER_RANDOM_FALLBACK, config)
    if not game.done and game.model_failures >= config.max_model_failures:
        game.result = "*"
        game.termination = "failure_cap"


def _legal_parsed_move(board: chess.Board, move_uci: str | None) -> chess.Move | None:
    if not move_uci:
        return None
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        return None
    return move if move in board.legal_moves else None


def _random_legal_move(game: GameState) -> str | None:
    legal = sorted(move.uci() for move in game.board.legal_moves)
    if not legal:
        return None
    return game.rng.choice(legal)


def _backfill_teacher_move(
    prompt: PromptArtifact,
    judgment: JudgmentArtifact,
    judge_engine: Any,
    config: SelfPlayConfig,
) -> JudgmentArtifact:
    try:
        board = chess.Board(prompt.fen)
        info = judge_engine.analyse(
            board,
            chess.engine.Limit(depth=int(config.judge_depth)),
        )
        pv = info.get("pv") or []
        teacher_move = pv[0].uci() if pv else None
    except Exception as exc:
        logger.warning("Teacher backfill failed for %s: %s", prompt.prompt_id, exc)
        return judgment
    if teacher_move is None:
        return judgment
    return replace(
        judgment,
        teacher_move_uci=teacher_move,
        metadata={**judgment.metadata, "teacher_move_source": "driver_backfill"},
    )


def _update_adjudication(
    game: GameState,
    judgment: JudgmentArtifact,
    mover_color: bool,
    config: SelfPlayConfig,
) -> None:
    if config.judge_depth <= 0 or config.adjudicate_plies <= 0:
        return
    move_eval_cp = judgment.metadata.get("stockfish_move_eval_cp")
    if not judgment.metadata.get("stockfish_scored") or move_eval_cp is None:
        game.adjudication_streak = 0
        return
    white_cp = float(move_eval_cp) if mover_color == chess.WHITE else -float(move_eval_cp)
    if abs(white_cp) < config.adjudicate_cp:
        game.adjudication_streak = 0
        return
    sign = 1 if white_cp > 0 else -1
    if sign == game.adjudication_sign:
        game.adjudication_streak += 1
    else:
        game.adjudication_sign = sign
        game.adjudication_streak = 1
    if game.adjudication_streak >= config.adjudicate_plies:
        game.result = "1-0" if sign > 0 else "0-1"
        game.termination = "adjudicated"


def _push_move(
    game: GameState,
    move: chess.Move,
    mover: str,
    config: SelfPlayConfig,
) -> None:
    ply = len(game.moves)
    game.positions.append(
        {
            "fen": game.board.fen(),
            "move_played_uci": move.uci(),
            "game_phase": game_phase(ply),
            "material_balance": material_balance(game.board),
            "ply": ply,
            "game_id": None,
            "mover": mover,
            "model_id": config.model_id,
            "run_id": config.run_id,
        }
    )
    game.board.push(move)
    game.moves.append(move.uci())

    outcome = game.board.outcome(claim_draw=True)
    if outcome is not None:
        game.result = outcome.result()
        game.termination = outcome.termination.name.lower()
    elif len(game.moves) >= config.max_plies:
        game.result = "*"
        game.termination = "max_plies"


def _finalize_game(game: GameState) -> None:
    payload = f"{game.start_fen}|{' '.join(game.moves)}".encode("utf-8")
    game.game_id = hashlib.sha256(payload).hexdigest()[:16]
    for position in game.positions:
        position["game_id"] = game.game_id
    for artifact in (*game.prompts, *game.rollouts, *game.judgments):
        artifact.metadata["game_id"] = game.game_id


def _artifact_id(config: SelfPlayConfig, game: GameState, ply: int) -> str:
    return f"selfplay-{config.run_id}-g{game.game_index:04d}-p{ply:03d}"


def _benchmark_example(prompt: PromptArtifact) -> BenchmarkExample:
    return BenchmarkExample(
        example_id=prompt.prompt_id,
        split="selfplay",
        task_type=SELF_PLAY_TASK_TYPE,
        fen=prompt.fen or "",
        prompt=_user_message(prompt),
        gold_answer="",
        metric_type="move_extraction",
    )


def _user_message(prompt: PromptArtifact) -> str:
    for message in prompt.messages:
        if message.role == "user":
            return message.content
    raise ValueError(f"prompt {prompt.prompt_id!r} has no user message")


def _write_outputs(
    games: list[GameState],
    config: SelfPlayConfig,
    *,
    judge_mode: str,
    judge_engine_name: str | None,
) -> SelfPlayResult:
    run_dir = Path(config.output_dir) / str(config.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    prompts = [prompt for game in games for prompt in game.prompts]
    rollouts = [rollout for game in games for rollout in game.rollouts]
    judgments = [judgment for game in games for judgment in game.judgments]
    positions = [position for game in games for position in game.positions]
    game_rows = [
        {
            "game_id": game.game_id,
            "opponent": game.opponent,
            "opponent_elo": game.opponent_elo,
            "model_side": "white" if game.model_side == chess.WHITE else "black",
            "result": game.result,
            "termination": game.termination,
            "n_plies": len(game.moves),
            "moves": list(game.moves),
            "start_fen": game.start_fen,
        }
        for game in games
    ]
    terminations = Counter(str(game.termination) for game in games)

    prompts_path = run_dir / "prompts.jsonl"
    rollouts_path = run_dir / "rollouts.jsonl"
    judgments_path = run_dir / "judgments.jsonl"
    games_path = run_dir / "games.jsonl"
    positions_path = run_dir / "positions.jsonl"
    manifest_path = run_dir / "manifest.json"

    write_jsonl(prompts_path, prompts)
    write_jsonl(rollouts_path, rollouts)
    write_jsonl(judgments_path, judgments)
    _write_dict_jsonl(games_path, game_rows)
    _write_dict_jsonl(positions_path, positions)

    manifest = {
        "schema_version": "artifact.v1",
        "artifact_type": "self_play_manifest",
        "run_id": config.run_id,
        "model_id": config.model_id,
        "model": config.model,
        "backend": config.backend,
        "output_dir": str(run_dir),
        "seed": config.seed,
        "temperature": config.temperature,
        "max_new_tokens": config.max_new_tokens,
        "game_count": len(games),
        "prompt_count": len(prompts),
        "rollout_count": len(rollouts),
        "judgment_count": len(judgments),
        "position_count": len(positions),
        "terminations": dict(sorted(terminations.items())),
        "opponent": {
            "mode": config.opponent,
            "self_play_fraction": config.self_play_fraction,
            "elo_min": config.opponent_elo_min,
            "elo_max": config.opponent_elo_max,
            "nodes": config.opponent_nodes,
        },
        "book": {
            "book_dir": config.book_dir,
            "book_plies": config.book_plies,
        },
        "limits": {
            "max_plies": config.max_plies,
            "max_model_failures": config.max_model_failures,
            "adjudicate_cp": config.adjudicate_cp,
            "adjudicate_plies": config.adjudicate_plies,
        },
        "judge": {
            "mode": judge_mode,
            "stockfish_path": config.stockfish_path,
            "stockfish_depth": config.judge_depth,
            "stockfish_engine_name": judge_engine_name,
            "chess960": False,
        },
        "outputs": {
            "prompts": str(prompts_path),
            "rollouts": str(rollouts_path),
            "judgments": str(judgments_path),
            "games": str(games_path),
            "positions": str(positions_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return SelfPlayResult(
        run_dir=run_dir,
        prompts_path=prompts_path,
        rollouts_path=rollouts_path,
        judgments_path=judgments_path,
        games_path=games_path,
        positions_path=positions_path,
        manifest_path=manifest_path,
        game_count=len(games),
        prompt_count=len(prompts),
        rollout_count=len(rollouts),
        judgment_count=len(judgments),
        position_count=len(positions),
        terminations=dict(terminations),
    )


def _write_dict_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


__all__ = [
    "BatchMoveGenerator",
    "ClientMoveGenerator",
    "GameState",
    "SelfPlayConfig",
    "SelfPlayResult",
    "TransformersMoveGenerator",
    "VllmMoveGenerator",
    "build_move_generator",
    "main",
    "run_self_play",
]


if __name__ == "__main__":
    raise SystemExit(main())
