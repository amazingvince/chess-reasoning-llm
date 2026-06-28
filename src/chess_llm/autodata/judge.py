"""Bootstrap rollout judging for parse and legality outcomes."""

from __future__ import annotations

import hashlib
from typing import Any

from chess_llm.artifacts.schemas import JudgmentArtifact, PromptArtifact, RolloutArtifact
from chess_llm.autodata.failure_buckets import (
    ILLEGAL_MOVE,
    LEGAL_UNSCORED,
    MISSING_FEN,
    PARSE_FAILURE,
)
from chess_llm.core.board import is_legal_move


def judge_rollout(
    prompt: PromptArtifact,
    rollout: RolloutArtifact,
    judgment_id: str | None = None,
    chess960: bool = False,
    metadata: dict[str, Any] | None = None,
) -> JudgmentArtifact:
    """Judge a rollout for parseability and move legality."""
    resolved_id = judgment_id or _judgment_id(prompt.prompt_id, rollout.rollout_id)
    base_metadata = dict(metadata or {})

    if not prompt.fen:
        return JudgmentArtifact(
            judgment_id=resolved_id,
            rollout_id=rollout.rollout_id,
            legal=None,
            failure_bucket=MISSING_FEN,
            feedback="Cannot judge rollout: missing FEN on prompt.",
            metadata=base_metadata,
        )

    move_uci = rollout.parsed_answer.move_uci
    if move_uci is None:
        detail = rollout.parsed_answer.parse_error or "no parsed move"
        return JudgmentArtifact(
            judgment_id=resolved_id,
            rollout_id=rollout.rollout_id,
            legal=False,
            failure_bucket=PARSE_FAILURE,
            feedback=f"Could not parse a move from model output: {detail}.",
            metadata=base_metadata,
        )

    if not is_legal_move(prompt.fen, move_uci, chess960=chess960):
        return JudgmentArtifact(
            judgment_id=resolved_id,
            rollout_id=rollout.rollout_id,
            legal=False,
            failure_bucket=ILLEGAL_MOVE,
            feedback=f"Parsed move {move_uci} is illegal in the prompt position.",
            metadata=base_metadata,
        )

    return JudgmentArtifact(
        judgment_id=resolved_id,
        rollout_id=rollout.rollout_id,
        legal=True,
        failure_bucket=LEGAL_UNSCORED,
        feedback=f"Parsed move {move_uci} is legal; engine regret is not scored yet.",
        metadata=base_metadata,
    )


def _judgment_id(prompt_id: str, rollout_id: str) -> str:
    payload = "\n".join([prompt_id, rollout_id]).encode("utf-8")
    return f"judgment-{hashlib.sha256(payload).hexdigest()[:16]}"
