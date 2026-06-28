"""JSONL writer for SFT training examples."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TextIO

from validation.validator import validate_example

logger = logging.getLogger(__name__)

# Log at most this many validation errors per writer instance
_MAX_LOGGED_ERRORS = 5


class JSONLWriter:
    """Write validated examples as one JSON object per line.

    Usage::

        with JSONLWriter("output/tier1/1.1_fen_to_board.jsonl") as w:
            w.write(example_dict)
    """

    def __init__(
        self,
        output_path: str,
        *,
        commit_on_close: bool = True,
        expected_task_id: str | None = None,
    ) -> None:
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO | None = None
        self._tmp_path: Path | None = None
        self._commit_on_close = commit_on_close
        self._expected_task_id = expected_task_id
        self._count = 0
        self._errors = 0

    def open(self) -> JSONLWriter:
        self._tmp_path = self.path.with_name(self.path.name + ".tmp")
        self._fh = open(self._tmp_path, "w", encoding="utf-8")
        return self

    def write(self, example: dict) -> bool:
        """Validate and write one example. Returns True if written."""
        if self._fh is None:
            raise RuntimeError("Writer not opened; use as context manager")

        passed, errors = validate_example(example)
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
                    self.path.name, task_id, errors,
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

    def __enter__(self) -> JSONLWriter:
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(commit=exc_type is None and self._commit_on_close)
