"""Stockfish-backed rollout judging."""

from __future__ import annotations

import hashlib
from typing import Any

import chess
import chess.engine

from chess_llm.artifacts.schemas import JudgmentArtifact, PromptArtifact, RolloutArtifact
from chess_llm.autodata.judge import judge_rollout


def judge_rollout_with_stockfish(
    prompt: PromptArtifact,
    rollout: RolloutArtifact,
    engine: Any,
    judgment_id: str | None = None,
    depth: int = 20,
    chess960: bool = False,
    metadata: dict[str, Any] | None = None,
) -> JudgmentArtifact:
    """Judge a rollout with parse/legality checks plus Stockfish regret."""
    resolved_id = judgment_id or _stockfish_judgment_id(prompt.prompt_id, rollout.rollout_id)
    base_metadata = {
        "judge": "stockfish",
        "stockfish_depth": int(depth),
        **dict(metadata or {}),
    }

    bootstrap = judge_rollout(
        prompt,
        rollout,
        judgment_id=resolved_id,
        chess960=chess960,
        metadata=base_metadata,
    )
    if bootstrap.legal is not True:
        return bootstrap

    move_uci = rollout.parsed_answer.move_uci
    if move_uci is None:
        return bootstrap

    try:
        board = chess.Board(prompt.fen, chess960=chess960)
        original_turn = board.turn
        move = chess.Move.from_uci(move_uci)

        best_info = engine.analyse(board, chess.engine.Limit(depth=int(depth)))
        teacher_move = _first_pv_move(best_info)
        best_eval = _score_to_cp(best_info.get("score"), original_turn)

        board.push(move)
        move_info = engine.analyse(board, chess.engine.Limit(depth=int(depth)))
        move_eval = _score_to_cp(move_info.get("score"), original_turn)
    except Exception as exc:
        fallback_metadata = dict(bootstrap.metadata)
        fallback_metadata["stockfish_scored"] = False
        fallback_metadata["stockfish_error"] = str(exc)
        return JudgmentArtifact(
            judgment_id=bootstrap.judgment_id,
            rollout_id=bootstrap.rollout_id,
            legal=bootstrap.legal,
            regret_cp=None,
            failure_bucket=bootstrap.failure_bucket,
            teacher_move_uci=None,
            feedback=f"{bootstrap.feedback} Stockfish scoring failed: {exc}",
            metadata=fallback_metadata,
        )

    if best_eval is None or move_eval is None:
        fallback_metadata = dict(bootstrap.metadata)
        fallback_metadata["stockfish_scored"] = False
        fallback_metadata["stockfish_best_eval_cp"] = best_eval
        fallback_metadata["stockfish_move_eval_cp"] = move_eval
        return JudgmentArtifact(
            judgment_id=bootstrap.judgment_id,
            rollout_id=bootstrap.rollout_id,
            legal=bootstrap.legal,
            regret_cp=None,
            failure_bucket=bootstrap.failure_bucket,
            teacher_move_uci=teacher_move,
            feedback=f"{bootstrap.feedback} Stockfish did not return complete scores.",
            metadata=fallback_metadata,
        )

    if teacher_move == move_uci:
        # The model played the engine's best move; any eval delta is
        # depth-parity noise and must not become a self-correction row.
        regret_cp = 0.0
    else:
        regret_cp = float(max(0, best_eval - move_eval))
    scored_metadata = dict(bootstrap.metadata)
    scored_metadata.update(
        {
            "stockfish_scored": True,
            "stockfish_best_eval_cp": best_eval,
            "stockfish_move_eval_cp": move_eval,
        }
    )

    return JudgmentArtifact(
        judgment_id=bootstrap.judgment_id,
        rollout_id=bootstrap.rollout_id,
        legal=True,
        regret_cp=regret_cp,
        failure_bucket=None,
        teacher_move_uci=teacher_move,
        feedback=(
            f"Stockfish scored parsed move {move_uci} with regret "
            f"{regret_cp:.1f} cp."
        ),
        metadata=scored_metadata,
    )


def _first_pv_move(info: dict[str, Any]) -> str | None:
    pv = info.get("pv") or []
    if not pv:
        return None
    return pv[0].uci()


def _score_to_cp(score: Any, color: chess.Color) -> int | None:
    if score is None:
        return None
    if hasattr(score, "pov"):
        score = score.pov(color)
    score_cp = score.score(mate_score=100000)
    return int(score_cp) if score_cp is not None else None


def _stockfish_judgment_id(prompt_id: str, rollout_id: str) -> str:
    payload = "\n".join([prompt_id, rollout_id, "stockfish"]).encode("utf-8")
    return f"judgment-{hashlib.sha256(payload).hexdigest()[:16]}"
