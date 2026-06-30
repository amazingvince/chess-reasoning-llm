"""Output writing and accounting for SFT data generation."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TextIO

from chess_llm.sft.settings import DEFAULT_CHESS960_RATIOS, DEFAULT_VOLUMES
from chess_llm.sft.validation import validate_example

logger = logging.getLogger(__name__)

_MAX_LOGGED_ERRORS = 5


class JSONLWriter:
    """Write validated examples as one JSON object per line."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        commit_on_close: bool = True,
        expected_task_id: str | None = None,
        validator: Callable[[dict], tuple[bool, list[str]]] = validate_example,
    ) -> None:
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO | None = None
        self._tmp_path: Path | None = None
        self._commit_on_close = commit_on_close
        self._expected_task_id = expected_task_id
        self._validator = validator
        self._count = 0
        self._errors = 0

    def open(self) -> "JSONLWriter":
        self._tmp_path = self.path.with_name(self.path.name + ".tmp")
        self._fh = self._tmp_path.open("w", encoding="utf-8", newline="\n")
        return self

    def write(self, example: dict) -> bool:
        """Validate and write one example. Returns True if written."""
        if self._fh is None:
            raise RuntimeError("Writer not opened; use as context manager")

        passed, errors = self._validator(example)
        if (
            self._expected_task_id is not None
            and example.get("task") != self._expected_task_id
        ):
            passed = False
            errors.append(
                "Task id mismatch: "
                f"expected {self._expected_task_id}, got {example.get('task')}"
            )
        if not passed:
            self._errors += 1
            if self._errors <= _MAX_LOGGED_ERRORS:
                task_id = example.get("task", example.get("task_id", "<unknown>"))
                logger.warning(
                    "Validation failed for %s (%s): %s",
                    self.path.name,
                    task_id,
                    errors,
                )
            return False

        self._fh.write(json.dumps(example, ensure_ascii=False) + "\n")
        self._count += 1
        return True

    def close(self, *, commit: bool | None = None) -> None:
        if commit is None:
            commit = self._commit_on_close
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self._tmp_path is not None:
            if commit and self._tmp_path.exists():
                self._tmp_path.replace(self.path)
            elif self._tmp_path.exists():
                self._tmp_path.unlink()
            self._tmp_path = None

    def commit(self) -> None:
        """Publish the temporary output file to the final path."""
        self.close(commit=True)

    def discard(self) -> None:
        """Remove the temporary output file without publishing it."""
        self.close(commit=False)

    @property
    def count(self) -> int:
        return self._count

    @property
    def error_count(self) -> int:
        return self._errors

    def __enter__(self) -> "JSONLWriter":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(commit=exc_type is None and self._commit_on_close)


class PipelineStats:
    """Accumulate counts per task and Chess960 flag during generation."""

    def __init__(
        self,
        *,
        volumes: Mapping[str, int] | None = None,
        chess960_ratios: Mapping[int, float] | None = None,
    ) -> None:
        self._volumes = dict(DEFAULT_VOLUMES if volumes is None else volumes)
        self._chess960_ratios = dict(
            DEFAULT_CHESS960_RATIOS
            if chess960_ratios is None
            else chess960_ratios
        )
        self._total: dict[str, int] = defaultdict(int)
        self._chess960: dict[str, int] = defaultdict(int)
        self._passed: dict[str, int] = defaultdict(int)
        self._failed: dict[str, int] = defaultdict(int)

    def record(
        self,
        task_id: str,
        is_chess960: bool = False,
        passed_validation: bool = True,
    ) -> None:
        self._total[task_id] += 1
        if is_chess960:
            self._chess960[task_id] += 1
        if passed_validation:
            self._passed[task_id] += 1
        else:
            self._failed[task_id] += 1

    def report(self) -> str:
        """Return a human-readable summary table."""
        lines = [
            f"{'Task':<30} {'Target':>8} {'Total':>8} {'Pass':>8} {'Fail':>8} {'960%':>6}",
            "-" * 76,
        ]
        for task_id in sorted(self._total):
            target = self._volumes.get(task_id, 0)
            total = self._total[task_id]
            passed = self._passed[task_id]
            failed = self._failed[task_id]
            c960_pct = (
                f"{self._chess960[task_id] / total * 100:.1f}%"
                if total > 0
                else "n/a"
            )
            lines.append(
                f"{task_id:<30} {target:>8} {total:>8} {passed:>8} {failed:>8} {c960_pct:>6}"
            )
        return "\n".join(lines)

    def verify_chess960_mix(self) -> dict[str, dict[str, float]]:
        """Return per-tier actual vs. target Chess960 ratios."""
        tier_totals: dict[int, int] = defaultdict(int)
        tier_960: dict[int, int] = defaultdict(int)

        for task_id, total in self._total.items():
            tier = int(task_id.split(".", 1)[0])
            tier_totals[tier] += total
            tier_960[tier] += self._chess960.get(task_id, 0)

        result: dict[str, dict[str, float]] = {}
        for tier in sorted(tier_totals):
            target = self._chess960_ratios.get(tier, 0.0)
            total = tier_totals[tier]
            actual = tier_960[tier] / total if total else 0.0
            result[f"tier_{tier}"] = {
                "target": target,
                "actual": round(actual, 4),
                "delta": round(actual - target, 4),
            }
        return result


__all__ = ["JSONLWriter", "PipelineStats"]
