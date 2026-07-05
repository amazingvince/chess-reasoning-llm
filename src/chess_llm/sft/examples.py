"""Helpers for SFT chat rows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from chess_llm.formats.prompts import SYSTEM_PROMPT


def build_sft_messages(
    *,
    user_prompt: str,
    assistant_content: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Build the standard three-message SFT chat sequence."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_content},
    ]


def build_sft_row(
    *,
    task: str,
    tier: int,
    fen: str,
    user_prompt: str,
    assistant_content: str,
    is_chess960: bool = False,
    metadata: Mapping[str, Any] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Build the JSONL row shape consumed by the trainer."""
    return {
        "task": task,
        "tier": int(tier),
        "fen": fen,
        "is_chess960": bool(is_chess960),
        "messages": build_sft_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            assistant_content=assistant_content,
        ),
        "metadata": dict(metadata or {}),
    }


@dataclass(frozen=True)
class SftExample:
    """A typed builder for one SFT training row."""

    task: str
    tier: int
    fen: str
    user_prompt: str
    assistant_content: str
    is_chess960: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    system_prompt: str = SYSTEM_PROMPT

    def to_dict(self) -> dict[str, Any]:
        return build_sft_row(
            task=self.task,
            tier=self.tier,
            fen=self.fen,
            user_prompt=self.user_prompt,
            assistant_content=self.assistant_content,
            is_chess960=self.is_chess960,
            metadata=self.metadata,
            system_prompt=self.system_prompt,
        )


def write_sft_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any] | SftExample],
) -> None:
    """Write SFT rows as JSONL, preserving order."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            payload = row.to_dict() if isinstance(row, SftExample) else dict(row)
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
