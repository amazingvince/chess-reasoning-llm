"""Read-only artifact loading for rollout review."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from chess_llm.artifacts.jsonl import read_jsonl
from chess_llm.artifacts.schemas import JudgmentArtifact, PromptArtifact, RolloutArtifact


class ArtifactLoadError(ValueError):
    """Raised when artifact files exist but cannot be parsed."""


@dataclass(frozen=True)
class JoinedRollout:
    """Prompt, rollout, and optional judgment joined for review."""

    prompt: PromptArtifact | None
    rollout: RolloutArtifact
    judgment: JudgmentArtifact | None

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt.to_dict() if self.prompt is not None else None,
            "rollout": self.rollout.to_dict(),
            "judgment": self.judgment.to_dict() if self.judgment is not None else None,
        }


@dataclass
class ArtifactRun:
    """One loaded artifact directory."""

    run_id: str
    artifact_dir: Path
    items: list[JoinedRollout]


def load_artifact_run(artifact_dir: str | Path, *, run_id: str | None = None) -> ArtifactRun:
    """Load and join prompt, rollout, and judgment JSONL files."""
    root = Path(artifact_dir)
    prompts_path = root / "prompts.jsonl"
    rollouts_path = root / "rollouts.jsonl"
    judgments_path = root / "judgments.jsonl"
    for path in [prompts_path, rollouts_path, judgments_path]:
        if not path.exists():
            raise FileNotFoundError(f"missing artifact file: {path}")

    try:
        prompts = {prompt.prompt_id: prompt for prompt in read_jsonl(prompts_path, PromptArtifact)}
        judgments = {
            judgment.rollout_id: judgment
            for judgment in read_jsonl(judgments_path, JudgmentArtifact)
        }
        items = [
            JoinedRollout(
                prompt=prompts.get(rollout.prompt_id),
                rollout=rollout,
                judgment=judgments.get(rollout.rollout_id),
            )
            for rollout in read_jsonl(rollouts_path, RolloutArtifact)
        ]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactLoadError(f"invalid artifact data in {root}: {exc}") from exc
    return ArtifactRun(run_id=run_id or _default_run_id(root), artifact_dir=root, items=items)


def _default_run_id(root: Path) -> str:
    resolved = str(root.resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    return f"{root.name}-{digest}"
