"""Unified FEN collection with deduplication and tagging."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from random import Random
from typing import Iterator

logger = logging.getLogger(__name__)


class FENPool:
    """In-memory FEN collection with metadata tags and sampling.

    Each FEN is stored once with associated tags (source, game_phase,
    piece_count, is_chess960, etc.).
    """

    def __init__(self) -> None:
        self._entries: dict[str, dict] = {}  # fen -> tags dict

    def add(self, fen: str, **tags) -> None:
        """Add a FEN with optional metadata tags. Skips duplicates."""
        if fen in self._entries:
            return
        self._entries[fen] = tags

    def __contains__(self, fen: str) -> bool:
        return fen in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def all_fens(self) -> list[str]:
        """Return all FENs in insertion order."""
        return list(self._entries.keys())

    def get_tags(self, fen: str) -> dict:
        """Return tags for a FEN, or empty dict."""
        return self._entries.get(fen, {})

    def sample(
        self,
        n: int,
        filters: dict | None = None,
        rng: Random | None = None,
    ) -> list[dict]:
        """Sample *n* entries, optionally filtered by tag values.

        Parameters
        ----------
        n : int
            Number of entries to sample.
        filters : dict, optional
            Tag key-value pairs to filter on. E.g. ``{"source": "lichess_games"}``.
        rng : Random, optional
            RNG instance for reproducibility.

        Returns
        -------
        list[dict]
            Each dict is ``{"fen": ..., **tags}``.
        """
        rng = rng or Random()
        candidates = []

        for fen, tags in self._entries.items():
            if filters:
                match = all(tags.get(k) == v for k, v in filters.items())
                if not match:
                    continue
            candidates.append({"fen": fen, **tags})

        if len(candidates) <= n:
            rng.shuffle(candidates)
            return candidates

        return rng.sample(candidates, n)

    def remove(self, fen: str) -> None:
        """Remove a FEN from the pool."""
        self._entries.pop(fen, None)

    def save(self, path: str) -> None:
        """Save pool to JSONL file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            for fen, tags in self._entries.items():
                obj = {"fen": fen, **tags}
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        logger.info("Saved FEN pool (%d entries) to %s", len(self), path)

    def load(self, path: str) -> None:
        """Load pool from JSONL file (additive — merges with existing)."""
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                fen = obj.pop("fen")
                self.add(fen, **obj)
        logger.info("Loaded FEN pool from %s (now %d entries)", path, len(self))
