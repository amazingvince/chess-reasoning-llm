"""Dataclass schemas for chess model improvement artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "artifact.v1"


def _metadata(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("metadata must be a dict")
    return dict(value)


def _require_artifact(payload: dict[str, Any], artifact_type: str) -> None:
    schema_version = payload.get("schema_version", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version!r}")
    found = payload.get("artifact_type")
    if found != artifact_type:
        raise ValueError(
            f"artifact_type must be {artifact_type!r}; found {found!r}"
        )


@dataclass
class ChatMessage:
    """One chat message in a prompt or training example."""

    role: str
    content: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.name is None:
            data.pop("name")
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChatMessage":
        return cls(
            role=str(payload["role"]),
            content=str(payload["content"]),
            name=payload.get("name"),
            metadata=_metadata(payload.get("metadata")),
        )


@dataclass
class ParsedAnswer:
    """Normalized answer parsed from raw model text."""

    raw_text: str = ""
    move_uci: str | None = None
    format_type: str = "none"
    parse_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)
    artifact_type: str = field(default="parsed_answer", init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParsedAnswer":
        _require_artifact(payload, "parsed_answer")
        return cls(
            raw_text=str(payload.get("raw_text", "")),
            move_uci=payload.get("move_uci"),
            format_type=str(payload.get("format_type", "none")),
            parse_error=payload.get("parse_error"),
            metadata=_metadata(payload.get("metadata")),
        )


@dataclass
class PromptArtifact:
    """A prompt prepared for evaluation, rollout, or data generation."""

    prompt_id: str
    messages: list[ChatMessage]
    fen: str | None = None
    task_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)
    artifact_type: str = field(default="prompt", init=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["messages"] = [message.to_dict() for message in self.messages]
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptArtifact":
        _require_artifact(payload, "prompt")
        return cls(
            prompt_id=str(payload["prompt_id"]),
            messages=[
                ChatMessage.from_dict(message)
                for message in payload.get("messages", [])
            ],
            fen=payload.get("fen"),
            task_type=payload.get("task_type"),
            metadata=_metadata(payload.get("metadata")),
        )


@dataclass
class RolloutArtifact:
    """A model response to a prompt, plus its parsed answer."""

    rollout_id: str
    prompt_id: str
    model_id: str
    raw_output: str
    parsed_answer: ParsedAnswer
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)
    artifact_type: str = field(default="rollout", init=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["parsed_answer"] = self.parsed_answer.to_dict()
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RolloutArtifact":
        _require_artifact(payload, "rollout")
        return cls(
            rollout_id=str(payload["rollout_id"]),
            prompt_id=str(payload["prompt_id"]),
            model_id=str(payload["model_id"]),
            raw_output=str(payload["raw_output"]),
            parsed_answer=ParsedAnswer.from_dict(payload["parsed_answer"]),
            metadata=_metadata(payload.get("metadata")),
        )


@dataclass
class JudgmentArtifact:
    """Verifier or teacher judgment for a rollout."""

    judgment_id: str
    rollout_id: str
    legal: bool | None = None
    regret_cp: float | None = None
    failure_bucket: str | None = None
    teacher_move_uci: str | None = None
    feedback: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)
    artifact_type: str = field(default="judgment", init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JudgmentArtifact":
        _require_artifact(payload, "judgment")
        regret = payload.get("regret_cp")
        return cls(
            judgment_id=str(payload["judgment_id"]),
            rollout_id=str(payload["rollout_id"]),
            legal=payload.get("legal"),
            regret_cp=float(regret) if regret is not None else None,
            failure_bucket=payload.get("failure_bucket"),
            teacher_move_uci=payload.get("teacher_move_uci"),
            feedback=payload.get("feedback"),
            metadata=_metadata(payload.get("metadata")),
        )


@dataclass
class PreferencePairArtifact:
    """Chosen/rejected pair for preference-style training."""

    pair_id: str
    prompt_id: str
    chosen_output: str
    rejected_output: str
    chosen_move_uci: str | None = None
    rejected_move_uci: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)
    artifact_type: str = field(default="preference_pair", init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreferencePairArtifact":
        _require_artifact(payload, "preference_pair")
        return cls(
            pair_id=str(payload["pair_id"]),
            prompt_id=str(payload["prompt_id"]),
            chosen_output=str(payload["chosen_output"]),
            rejected_output=str(payload["rejected_output"]),
            chosen_move_uci=payload.get("chosen_move_uci"),
            rejected_move_uci=payload.get("rejected_move_uci"),
            reason=payload.get("reason"),
            metadata=_metadata(payload.get("metadata")),
        )


@dataclass
class FeedbackDistillationArtifact:
    """Training item for feedback-conditioned self-distillation."""

    example_id: str
    prompt_id: str
    student_output: str
    feedback: str
    teacher_output: str
    target_output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)
    artifact_type: str = field(default="feedback_distillation", init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> "FeedbackDistillationArtifact":
        _require_artifact(payload, "feedback_distillation")
        return cls(
            example_id=str(payload["example_id"]),
            prompt_id=str(payload["prompt_id"]),
            student_output=str(payload["student_output"]),
            feedback=str(payload["feedback"]),
            teacher_output=str(payload["teacher_output"]),
            target_output=str(payload["target_output"]),
            metadata=_metadata(payload.get("metadata")),
        )
