"""Canonical FEN pool with deduplication, tags, sampling, and JSONL IO."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from random import Random
from typing import Any

from chess_llm.sft.context import normalize_chess_variant_metadata, raw_fen_identity_key

logger = logging.getLogger(__name__)


class FENPool:
    """In-memory FEN collection keyed by canonical board identity."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}

    def identity_key(self, fen: str, **tags: Any) -> str:
        """Return the canonical identity key this pool uses for *fen*."""
        return raw_fen_identity_key({"fen": fen, **tags})

    def add(self, fen: str, **tags: Any) -> None:
        """Add a FEN with optional metadata tags, skipping canonical duplicates."""
        row = normalize_chess_variant_metadata({"fen": fen, **tags})
        normalized_tags = {
            key: value
            for key, value in row.items()
            if key not in {"fen", "chess960_id"} and not (key == "metadata" and not value)
        }
        key = raw_fen_identity_key(row)
        if key in self._entries:
            _merge_tags(self._entries[key]["tags"], normalized_tags)
            return
        self._entries[key] = {
            "fen": fen,
            "tags": normalized_tags,
        }

    def __contains__(self, fen: str) -> bool:
        return self.identity_key(fen) in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def all_fens(self) -> list[str]:
        """Return stored FEN strings in insertion order."""
        return [entry["fen"] for entry in self._entries.values()]

    def all_rows(self) -> list[dict]:
        """Return stored FEN rows with their original variant-aware tags."""
        return [
            {"fen": entry["fen"], **entry["tags"]}
            for entry in self._entries.values()
        ]

    def get_tags(self, fen: str) -> dict:
        """Return a copy of tags for the canonical FEN identity."""
        entry = self._entries.get(self.identity_key(fen))
        if entry is None:
            return {}
        return dict(entry["tags"])

    def sample(
        self,
        n: int,
        filters: dict | None = None,
        rng: Random | None = None,
    ) -> list[dict]:
        """Sample entries, optionally filtered by tag values."""
        rng = rng or Random()
        candidates: list[dict] = []

        for entry in self._entries.values():
            tags = entry["tags"]
            if filters and not all(tags.get(key) == value for key, value in filters.items()):
                continue
            candidates.append({"fen": entry["fen"], **tags})

        if len(candidates) <= n:
            rng.shuffle(candidates)
            return candidates
        return rng.sample(candidates, n)

    def remove(self, fen: str) -> None:
        """Remove a FEN by canonical identity."""
        self._entries.pop(self.identity_key(fen), None)

    def save(self, path: str | Path) -> None:
        """Save pool rows to a JSONL file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as fh:
            for entry in self._entries.values():
                fh.write(
                    json.dumps(
                        {"fen": entry["fen"], **entry["tags"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        logger.info("Saved FEN pool (%d entries) to %s", len(self), output_path)

    def load(self, path: str | Path) -> None:
        """Load pool rows from a JSONL file, merging into existing entries."""
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                fen = row.pop("fen")
                self.add(fen, **row)
        logger.info("Loaded FEN pool from %s (now %d entries)", path, len(self))


__all__ = ["FENPool"]


def _merge_tags(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    """Merge non-conflicting later tags into an existing canonical entry."""
    for key, value in incoming.items():
        if key == "metadata" and isinstance(value, dict):
            current = existing.get("metadata")
            if not isinstance(current, dict):
                existing["metadata"] = dict(value)
            else:
                for meta_key, meta_value in value.items():
                    if meta_key == "is_chess960" and bool(meta_value):
                        current[meta_key] = True
                    elif meta_key == "chess960_id" and meta_value is not None:
                        current.setdefault(meta_key, meta_value)
                    else:
                        current.setdefault(meta_key, meta_value)
            continue
        if key == "is_chess960" and bool(value):
            existing[key] = True
            continue
        if key == "chess960_id" and value is not None:
            existing.setdefault(key, value)
            continue
        existing.setdefault(key, value)
