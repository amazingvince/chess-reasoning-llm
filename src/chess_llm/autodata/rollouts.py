"""Build rollout artifacts from raw model outputs."""

from __future__ import annotations

import hashlib
from typing import Any

from chess_llm.artifacts.schemas import PromptArtifact, RolloutArtifact
from chess_llm.formats.answers import parse_answer


def build_rollout(
    prompt: PromptArtifact,
    model_id: str,
    raw_output: str,
    rollout_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RolloutArtifact:
    """Create a rollout artifact and parse its move answer."""
    resolved_id = rollout_id or _rollout_id(prompt.prompt_id, model_id, raw_output)
    return RolloutArtifact(
        rollout_id=resolved_id,
        prompt_id=prompt.prompt_id,
        model_id=model_id,
        raw_output=raw_output,
        parsed_answer=parse_answer(raw_output),
        metadata=dict(metadata or {}),
    )


def _rollout_id(prompt_id: str, model_id: str, raw_output: str) -> str:
    payload = "\n".join([prompt_id, model_id, raw_output]).encode("utf-8")
    return f"rollout-{hashlib.sha256(payload).hexdigest()[:16]}"
