"""Source 4: Lichess Chess Position Evaluations.

Stream the 40GB dataset of 845M evaluation rows, filter by depth,
validate FEN + best move, dedup via SQLite (highest depth per FEN),
and partition by use case.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Iterator

import chess
from datasets import load_dataset

from config.settings import HF_DATASETS, ANNOTATIONS_DIR

logger = logging.getLogger(__name__)

_DEDUP_DB = ANNOTATIONS_DIR / "evals_dedup.db"


def _init_dedup_db(db_path: Path) -> sqlite3.Connection:
    """Create or open the dedup SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evals (
            fen TEXT PRIMARY KEY,
            best_move TEXT,
            pv_line TEXT,
            depth INTEGER,
            knodes INTEGER,
            cp INTEGER,
            mate INTEGER
        )
    """)
    conn.commit()
    return conn


def stream_evals(
    min_depth: int = 20,
    max_rows: int | None = None,
    dedup_db_path: Path | None = None,
) -> Iterator[dict]:
    """Stream position evaluations, filtering by depth and deduplicating.

    Uses SQLite WAL-mode for dedup: ``INSERT OR REPLACE`` keeping the
    highest-depth row per FEN.

    Yields::

        {fen, best_move, pv_line, depth, cp, mate}
    """
    db_path = dedup_db_path or _DEDUP_DB
    conn = _init_dedup_db(db_path)

    ds = load_dataset(
        HF_DATASETS["lichess_evals"], split="train", streaming=True
    )
    count = 0
    batch: list[tuple] = []
    # In-memory tracker covers the current unflushed batch so that
    # duplicates within the same batch are caught immediately.
    seen: dict[str, int] = {}  # fen -> best depth seen so far

    for row in ds:
        if max_rows is not None and count >= max_rows:
            break

        depth = row.get("depth", 0)
        if depth < min_depth:
            continue

        fen = row.get("fen", "")
        line = row.get("line", "")
        if not fen or not line:
            continue

        best_move = line.split()[0]

        # Validate FEN and best move
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(best_move)
            if move not in board.legal_moves:
                continue
        except (ValueError, TypeError):
            continue

        cp = row.get("cp")
        mate = row.get("mate")
        knodes = row.get("knodes", 0)

        # Dedup: check in-memory tracker first (covers unflushed batch),
        # then fall back to SQLite for earlier runs / flushed batches.
        prev_depth = seen.get(fen)
        if prev_depth is not None:
            if prev_depth >= depth:
                continue
        else:
            existing = conn.execute(
                "SELECT depth FROM evals WHERE fen = ?", (fen,)
            ).fetchone()
            if existing and existing[0] >= depth:
                continue

        seen[fen] = depth
        batch.append((fen, best_move, line, depth, knodes, cp, mate))

        if len(batch) >= 10_000:
            _flush_batch(conn, batch)
            batch.clear()
            # After flushing, the DB is authoritative; clear in-memory
            # tracker to bound memory usage.
            seen.clear()

        yield {
            "fen": fen,
            "best_move": best_move,
            "pv_line": line,
            "depth": depth,
            "cp": cp,
            "mate": mate,
        }
        count += 1

    if batch:
        _flush_batch(conn, batch)
    conn.close()


def _flush_batch(conn: sqlite3.Connection, batch: list[tuple]) -> None:
    """Insert or replace rows keeping highest depth per FEN."""
    conn.executemany(
        """
        INSERT INTO evals (fen, best_move, pv_line, depth, knodes, cp, mate)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fen) DO UPDATE SET
            best_move = excluded.best_move,
            pv_line   = excluded.pv_line,
            depth     = excluded.depth,
            knodes    = excluded.knodes,
            cp        = excluded.cp,
            mate      = excluded.mate
        WHERE excluded.depth > evals.depth
           OR (excluded.depth = evals.depth AND excluded.knodes > evals.knodes)
        """,
        batch,
    )
    conn.commit()


def partition_evals(
    evals: Iterator[dict],
) -> dict[str, list[dict]]:
    """Partition filtered evals into task-specific buckets.

    Returns::

        {mate, high_eval, balanced, endgame, best_move}
    """
    partitions: dict[str, list[dict]] = {
        "mate": [],
        "high_eval": [],
        "balanced": [],
        "endgame": [],
        "best_move": [],
    }

    for row in evals:
        board = chess.Board(row["fen"])
        piece_count = len(board.piece_map())

        if row.get("mate") is not None:
            partitions["mate"].append(row)

        cp = row.get("cp")
        if cp is not None:
            if abs(cp) > 200:
                partitions["high_eval"].append(row)
            elif abs(cp) < 100:
                partitions["balanced"].append(row)

        if piece_count <= 10:
            partitions["endgame"].append(row)

        if row.get("depth", 0) >= 30:
            partitions["best_move"].append(row)

    return partitions
