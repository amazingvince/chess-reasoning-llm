"""Filesystem helpers for generated SFT artifacts."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


IGNORED_JSONL_PATH_PARTS = frozenset({".training_cache", "__pycache__"})


def is_visible_artifact_path(path: str | Path, root: str | Path) -> bool:
    """Return True when *path* is not inside hidden/cache artifact directories."""
    candidate = Path(path)
    base = Path(root)
    try:
        relative = candidate.relative_to(base)
    except ValueError:
        relative = candidate
    return not any(
        part.startswith(".") or part in IGNORED_JSONL_PATH_PARTS
        for part in relative.parts
    )


def iter_jsonl_artifacts(root: str | Path, *, recursive: bool = True) -> Iterator[Path]:
    """Yield generated JSONL artifacts while skipping hidden/cache directories."""
    base = Path(root)
    paths = base.rglob("*.jsonl") if recursive else base.glob("*.jsonl")
    for path in sorted(paths):
        if path.is_file() and is_visible_artifact_path(path, base):
            yield path


__all__ = [
    "IGNORED_JSONL_PATH_PARTS",
    "is_visible_artifact_path",
    "iter_jsonl_artifacts",
]
