"""Completeness checks for generated tier JSONL outputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from chess_llm.sft.settings import DEFAULT_VOLUMES


def expected_task_path(output_dir: str | Path, task_id: str) -> Path:
    """Return the expected JSONL path for a generated task."""
    tier = task_id.split(".", 1)[0]
    return Path(output_dir) / f"tier{tier}" / f"{task_id}.jsonl"


def audit_output_completeness(
    output_dir: str | Path,
    expected_volume_override: int | None = None,
    *,
    volumes: Mapping[str, int] | None = None,
    tiers: Iterable[int] | None = None,
    task_ids: Iterable[str] | None = None,
) -> dict[str, str]:
    """Return missing/underfilled output files keyed by relative path."""
    base = Path(output_dir)
    expected_volumes = filter_task_volumes(
        DEFAULT_VOLUMES if volumes is None else volumes,
        tiers=tiers,
        task_ids=task_ids,
    )
    issues: dict[str, str] = {}
    for task_id, configured_target in sorted(expected_volumes.items()):
        expected = (
            expected_volume_override
            if expected_volume_override is not None
            else configured_target
        )
        path = expected_task_path(base, task_id)
        rel_path = str(path.relative_to(base))
        if not path.exists():
            issues[rel_path] = f"missing expected task file (target {expected})"
            continue

        with path.open(encoding="utf-8") as fh:
            count = sum(1 for line in fh if line.strip())

        if count < expected:
            issues[rel_path] = f"underfilled: {count} / {expected}"

    return issues


def filter_task_volumes(
    volumes: Mapping[str, int],
    *,
    tiers: Iterable[int] | None = None,
    task_ids: Iterable[str] | None = None,
) -> dict[str, int]:
    """Return expected task volumes narrowed to selected tiers/tasks."""
    selected = dict(volumes)
    if tiers is not None:
        tier_set = {str(tier) for tier in tiers}
        selected = {
            task_id: target
            for task_id, target in selected.items()
            if task_id.split(".", 1)[0] in tier_set
        }
    if task_ids is not None:
        task_set = set(task_ids)
        selected = {
            task_id: target
            for task_id, target in selected.items()
            if task_id in task_set
        }
    return selected


__all__ = ["audit_output_completeness", "expected_task_path", "filter_task_volumes"]
