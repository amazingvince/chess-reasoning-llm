"""Read and write ordered JSONL artifact files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Protocol, TypeVar


class JsonSerializable(Protocol):
    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""


T = TypeVar("T")


def write_jsonl(path: str | Path, rows: Iterable[JsonSerializable]) -> None:
    """Write rows as JSONL, preserving input order."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path, artifact_cls: type[T]) -> Iterator[T]:
    """Read JSONL rows and construct artifacts with ``artifact_cls.from_dict``."""
    from_dict = getattr(artifact_cls, "from_dict", None)
    if from_dict is None:
        raise TypeError("artifact_cls must define from_dict")

    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield from_dict(json.loads(line))
