"""Completeness checks for generated tier JSONL outputs."""

from __future__ import annotations

from pathlib import Path

from config.settings import VOLUMES


def expected_task_path(output_dir: str | Path, task_id: str) -> Path:
    """Return the expected JSONL path for a generated task."""
    tier = task_id.split(".", 1)[0]
    return Path(output_dir) / f"tier{tier}" / f"{task_id}.jsonl"


def audit_output_completeness(
    output_dir: str | Path,
    expected_volume_override: int | None = None,
) -> dict[str, str]:
    """Return missing/underfilled output files keyed by relative path."""
    base = Path(output_dir)
    issues: dict[str, str] = {}
    for task_id, configured_target in sorted(VOLUMES.items()):
        expected = expected_volume_override if expected_volume_override is not None else configured_target
        path = expected_task_path(base, task_id)
        rel_path = str(path.relative_to(base))
        if not path.exists():
            issues[rel_path] = f"missing expected task file (target {expected})"
            continue

        with open(path, encoding="utf-8") as fh:
            count = sum(1 for line in fh if line.strip())

        if count < expected:
            issues[rel_path] = f"underfilled: {count} / {expected}"

    return issues
