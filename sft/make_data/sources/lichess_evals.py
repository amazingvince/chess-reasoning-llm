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


def _normalize_pv_line(
    fen: str,
    line: str,
    chess960: bool = False,
) -> str:
    """Normalize every UCI move in a PV using python-chess parsing.

    This converts king-to-rook castling notation from engine datasets into
    the canonical UCI strings used by generated labels.
    """
    board = chess.Board(fen, chess960=chess960)
    normalized: list[str] = []
    for token in line.split():
        move = board.parse_uci(token)
        normalized.append(move.uci())
        board.push(move)
    return " ".join(normalized)


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

    Uses a two-phase approach:

    1. **Collect** — iterate the source, validate, and keep only the
       highest-depth row per FEN (checked against both the in-memory
       ``best`` dict and the on-disk SQLite DB from previous runs).
    2. **Yield** — emit the deduplicated results and flush to SQLite
       for cross-run persistence.

    This guarantees every FEN is yielded at most once (with the
    highest depth seen in the current scan).

    Yields::

        {fen, best_move, pv_line, depth, cp, mate}
    """
    db_path = dedup_db_path or _DEDUP_DB
    conn = _init_dedup_db(db_path)

    ds = load_dataset(
        HF_DATASETS["lichess_evals"], split="train", streaming=True
    )

    # Phase 1: collect, keeping only the highest-depth row per FEN.
    best: dict[str, dict] = {}  # fen -> row dict
    count = 0  # valid rows consumed from the source

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

        # Validate FEN and normalize the whole PV.  Lichess/Stockfish may
        # output king-to-rook castling notation; python-chess normalizes it.
        try:
            pv_line = _normalize_pv_line(fen, line)
            best_move = pv_line.split()[0]
        except (ValueError, TypeError):
            continue

        cp = row.get("cp")
        mate = row.get("mate")
        knodes = row.get("knodes", 0)
        count += 1

        # Dedup against SQLite (previous runs) — same tie-break as _flush_batch
        existing = conn.execute(
            "SELECT depth, knodes FROM evals WHERE fen = ?", (fen,)
        ).fetchone()
        if existing:
            ex_depth, ex_knodes = existing
            if ex_depth > depth:
                continue
            if ex_depth == depth and (ex_knodes or 0) >= knodes:
                continue

        # Dedup against current run's best (same tie-break as _flush_batch)
        prev = best.get(fen)
        if prev is not None:
            if prev["depth"] > depth:
                continue
            if prev["depth"] == depth and prev["knodes"] >= knodes:
                continue

        best[fen] = {
            "fen": fen,
            "best_move": best_move,
            "pv_line": pv_line,
            "depth": depth,
            "knodes": knodes,
            "cp": cp,
            "mate": mate,
        }

    # Phase 2: flush to SQLite for cross-run persistence, then yield.
    batch = [
        (r["fen"], r["best_move"], r["pv_line"], r["depth"],
         r["knodes"], r["cp"], r["mate"])
        for r in best.values()
    ]
    if batch:
        _flush_batch(conn, batch)

    if best:
        conn.close()
        yield from best.values()
        return

    # Re-run: all rows already in DB at equal-or-better quality.
    # Yield from the persisted DB instead of returning nothing.
    logger.info("No new evals; yielding from dedup DB")
    query = "SELECT fen, best_move, pv_line, depth, knodes, cp, mate FROM evals WHERE depth >= ?"
    params: list = [min_depth]
    if max_rows is not None:
        query += " LIMIT ?"
        params.append(max_rows)
    for row in conn.execute(query, params):
        fen = row[0]
        pv_line = row[2]
        # Normalize castling notation from DB (may have old king-to-rook data)
        try:
            pv_line = _normalize_pv_line(fen, pv_line)
            best_move = pv_line.split()[0]
        except (ValueError, TypeError):
            continue
        yield {
            "fen": fen, "best_move": best_move, "pv_line": pv_line,
            "depth": row[3], "knodes": row[4], "cp": row[5], "mate": row[6],
        }
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
