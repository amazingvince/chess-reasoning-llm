"""Stockfish annotation with Position Evals cache.

Uses an SQLite cache so annotations persist across pipeline runs.
The Lichess Position Evaluations dataset can pre-populate the cache,
so live Stockfish is only called for uncached FENs.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Iterator

import chess

from sources.stockfish_engine import StockfishWrapper

logger = logging.getLogger(__name__)


class BatchAnnotator:
    """Annotate positions with Stockfish, using a persistent cache.

    The cache is an SQLite database keyed on FEN. The Lichess Position
    Evaluations dataset pre-populates it, so live Stockfish is only
    called for FENs not found in the cache (e.g., Chess960 positions,
    custom endgames).

    Parameters
    ----------
    stockfish : StockfishWrapper
        An opened Stockfish engine wrapper.
    cache_path : str
        Path to the SQLite annotation cache.
    """

    def __init__(
        self,
        stockfish: StockfishWrapper | None = None,
        cache_path: str | None = None,
    ) -> None:
        self.stockfish = stockfish
        self.cache_path = Path(cache_path) if cache_path else None
        self._conn: sqlite3.Connection | None = None

        if self.cache_path:
            self._init_cache()

    def _init_cache(self) -> None:
        """Create or open the annotation cache."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.cache_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                fen TEXT PRIMARY KEY,
                cp INTEGER,
                mate INTEGER,
                best_move TEXT,
                pv_line TEXT,
                depth INTEGER
            )
        """)
        self._conn.commit()

    def preload_from_evals(self, evals: Iterator[dict]) -> int:
        """Pre-populate cache from Position Evals dataset rows.

        Each row must have: fen, cp, mate, best_move, pv_line, depth.
        Returns the number of rows inserted.
        """
        if self._conn is None:
            raise RuntimeError("No cache path configured")

        batch: list[tuple] = []
        count = 0

        for row in evals:
            batch.append((
                row["fen"],
                row.get("cp"),
                row.get("mate"),
                row.get("best_move", ""),
                row.get("pv_line", ""),
                row.get("depth", 0),
            ))
            if len(batch) >= 10_000:
                self._flush_preload(batch)
                count += len(batch)
                batch.clear()

        if batch:
            self._flush_preload(batch)
            count += len(batch)

        logger.info("Preloaded %d annotations into cache", count)
        return count

    def _flush_preload(self, batch: list[tuple]) -> None:
        self._conn.executemany(
            """
            INSERT OR IGNORE INTO annotations
                (fen, cp, mate, best_move, pv_line, depth)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        self._conn.commit()

    def annotate(self, fen: str, depth: int | None = None) -> dict:
        """Get annotation for a single FEN.

        Checks cache first, then falls back to live Stockfish.
        Returns ``{cp, mate, best_move, pv_line}``.
        """
        # Check cache
        cached = self._cache_lookup(fen)
        if cached is not None:
            return cached

        # Live Stockfish
        if self.stockfish is None:
            return {"cp": None, "mate": None, "best_move": None, "pv_line": ""}

        board = chess.Board(fen)
        result = self.stockfish.evaluate(board, depth)

        # Store in cache
        self._cache_store(fen, result, depth)
        return result

    def annotate_batch(
        self, fens: list[str], depth: int | None = None
    ) -> Iterator[dict]:
        """Annotate multiple FENs."""
        for fen in fens:
            yield self.annotate(fen, depth)

    def _cache_lookup(self, fen: str) -> dict | None:
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT cp, mate, best_move, pv_line FROM annotations WHERE fen = ?",
            (fen,),
        ).fetchone()
        if row is None:
            return None
        return {
            "cp": row[0],
            "mate": row[1],
            "best_move": row[2],
            "pv_line": row[3],
        }

    def _cache_store(self, fen: str, result: dict, depth: int | None) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            """
            INSERT OR REPLACE INTO annotations
                (fen, cp, mate, best_move, pv_line, depth)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                fen,
                result.get("cp"),
                result.get("mate"),
                result.get("best_move", ""),
                result.get("pv_line", ""),
                depth or 0,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
