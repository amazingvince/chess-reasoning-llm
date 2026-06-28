import chess
import chess.engine

from chess_llm.artifacts.schemas import ChatMessage, PromptArtifact
from chess_llm.autodata.failure_buckets import PARSE_FAILURE
from chess_llm.autodata.rollouts import build_rollout
from chess_llm.autodata.stockfish_judge import judge_rollout_with_stockfish


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class FakeStockfish:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def analyse(self, board: chess.Board, limit: chess.engine.Limit) -> dict:
        self.calls.append((board.fen(), limit.depth))
        if len(board.move_stack) == 0:
            return {
                "score": chess.engine.PovScore(chess.engine.Cp(35), chess.WHITE),
                "pv": [chess.Move.from_uci("e2e4")],
            }
        if board.peek().uci() == "d2d4":
            return {"score": chess.engine.PovScore(chess.engine.Cp(10), chess.WHITE)}
        return {"score": chess.engine.PovScore(chess.engine.Cp(35), chess.WHITE)}


def _prompt() -> PromptArtifact:
    return PromptArtifact(
        prompt_id="planning_00000",
        messages=[ChatMessage(role="user", content="Choose a move.")],
        fen=STARTING_FEN,
        task_type="best_move",
    )


def test_stockfish_judge_adds_teacher_move_and_regret_for_legal_rollout():
    prompt = _prompt()
    rollout = build_rollout(prompt, "model-a", "<move>d2d4</move>", "rollout-1")
    engine = FakeStockfish()

    judgment = judge_rollout_with_stockfish(
        prompt,
        rollout,
        engine,
        depth=18,
        judgment_id="judgment-1",
    )

    assert judgment.judgment_id == "judgment-1"
    assert judgment.rollout_id == "rollout-1"
    assert judgment.legal is True
    assert judgment.failure_bucket is None
    assert judgment.teacher_move_uci == "e2e4"
    assert judgment.regret_cp == 25.0
    assert judgment.metadata["judge"] == "stockfish"
    assert judgment.metadata["stockfish_depth"] == 18
    assert judgment.metadata["stockfish_best_eval_cp"] == 35
    assert judgment.metadata["stockfish_move_eval_cp"] == 10
    assert [depth for _, depth in engine.calls] == [18, 18]


def test_stockfish_judge_preserves_parse_failure_without_engine_analysis():
    prompt = _prompt()
    rollout = build_rollout(prompt, "model-a", "I considered e2e4 and d2d4.", "rollout-1")
    engine = FakeStockfish()

    judgment = judge_rollout_with_stockfish(prompt, rollout, engine)

    assert judgment.legal is False
    assert judgment.failure_bucket == PARSE_FAILURE
    assert judgment.regret_cp is None
    assert judgment.teacher_move_uci is None
    assert engine.calls == []
