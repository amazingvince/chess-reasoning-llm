"""Lichess position-eval dataset parsing, streaming, and partitioning."""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path

import chess

from chess_llm.sft.context import board_from_raw
from chess_llm.sft.settings import (
    DEFAULT_HF_CACHE_DIR,
    DEFAULT_HF_DATASETS,
    DEFAULT_MIN_DEPTH_TRAINING,
    SftDataSettings,
)
from chess_llm.sft.sources._streaming import (
    close_streaming_dataset,
    streaming_shutdown_wait_seconds,
)

EVAL_PERSPECTIVE = "white"
logger = logging.getLogger(__name__)

DatasetLoader = Callable[..., Iterable[Mapping]]

# Flush accepted rows to the dedup DB every this many entries so a mid-stream
# failure loses at most one partial batch.
_FLUSH_INTERVAL = 5000
# Seed for the deterministic shuffle applied when yielding cached evals.
_CACHE_SHUFFLE_SEED = 0


def normalize_pv_line(fen: str, line: str, *, chess960: bool = False) -> str:
    """Normalize a principal variation into canonical UCI moves."""
    board = chess.Board(fen, chess960=chess960)
    if not board.is_valid():
        raise ValueError(f"invalid FEN: {fen}")
    normalized: list[str] = []
    for token in line.split():
        move = board.parse_uci(token)
        normalized.append(move.uci())
        board.push(move)
    return " ".join(normalized)


def parse_lichess_eval_row(row: dict, *, chess960: bool = False) -> dict | None:
    """Parse one Lichess eval row and document its score perspective.

    The source ``cp`` and ``mate`` values are treated downstream as White POV:
    positive means White is better or mating, negative means Black is better
    or mating.  Preserve that contract explicitly on every parsed row.
    """
    fen = row.get("fen", "")
    line = row.get("line") or row.get("pv_line", "")
    if not fen or not line:
        return None

    try:
        pv_line = normalize_pv_line(fen, line, chess960=chess960)
    except (ValueError, TypeError):
        return None

    if not pv_line:
        return None

    return {
        "fen": fen,
        "best_move": pv_line.split()[0],
        "pv_line": pv_line,
        "depth": row.get("depth", 0),
        "knodes": row.get("knodes", 0),
        "cp": row.get("cp"),
        "mate": row.get("mate"),
        "eval_perspective": EVAL_PERSPECTIVE,
    }


def stream_evals(
    min_depth: int = DEFAULT_MIN_DEPTH_TRAINING,
    max_rows: int | None = None,
    dedup_db_path: Path | None = None,
    *,
    dataset_loader: DatasetLoader | None = None,
    dataset_name: str = DEFAULT_HF_DATASETS["lichess_evals"],
) -> Iterator[dict]:
    """Stream, validate, and deduplicate Lichess position-eval rows.

    ``max_rows`` caps valid parsed source rows consumed before yielding the
    best row per FEN.  The SQLite cache persists higher-quality rows across
    runs and is used as a fallback source when a later scan finds no better
    rows.
    """
    if max_rows is not None and max_rows <= 0:
        return

    db_path = dedup_db_path or _default_dedup_db_path()
    conn = _init_dedup_db(db_path)
    uses_default_loader = dataset_loader is None
    loader = dataset_loader or _default_dataset_loader

    best: dict[str, dict] = {}
    valid_count = 0
    flushed_any = False

    try:
        try:
            rows = loader(dataset_name, split="train", streaming=True)
            row_iter = iter(rows)
            try:
                for row in row_iter:
                    if max_rows is not None and valid_count >= max_rows:
                        break

                    parsed = parse_lichess_eval_row(dict(row))
                    if parsed is None:
                        continue

                    depth = _coerce_int(parsed.get("depth"), default=0)
                    knodes = _coerce_int(parsed.get("knodes"), default=0)
                    if depth < min_depth:
                        continue

                    valid_count += 1
                    parsed["depth"] = depth
                    parsed["knodes"] = knodes
                    fen = parsed["fen"]

                    existing = conn.execute(
                        "SELECT depth, knodes FROM evals WHERE fen = ?",
                        (fen,),
                    ).fetchone()
                    if existing is not None:
                        ex_depth, ex_knodes = existing
                        if ex_depth > depth:
                            continue
                        if ex_depth == depth and (ex_knodes or 0) >= knodes:
                            continue

                    prev = best.get(fen)
                    if prev is not None:
                        if prev["depth"] > depth:
                            continue
                        if prev["depth"] == depth and prev["knodes"] >= knodes:
                            continue

                    best[fen] = parsed
                    if len(best) >= _FLUSH_INTERVAL:
                        _flush_best(conn, best)
                        best.clear()
                        flushed_any = True
            finally:
                close_streaming_dataset(
                    row_iter,
                    rows,
                    shutdown_wait_seconds=(
                        streaming_shutdown_wait_seconds() if uses_default_loader else 0.0
                    ),
                )
        except Exception:
            logger.exception(
                "Eval stream scan failed mid-stream after %d accepted rows; "
                "yielding rows already flushed to %s",
                valid_count,
                db_path,
            )

        if best:
            _flush_best(conn, best)
            flushed_any = True
        if not flushed_any:
            logger.info("No new evals; yielding from dedup DB")

        yield from _iter_cached_evals(conn, min_depth=min_depth, max_rows=max_rows)
    finally:
        conn.close()


def _flush_best(conn: sqlite3.Connection, best: dict[str, dict]) -> None:
    _flush_batch(
        conn,
        [
            (
                row["fen"],
                row["best_move"],
                row["pv_line"],
                row["depth"],
                row["knodes"],
                row["cp"],
                row["mate"],
            )
            for row in best.values()
        ],
    )


def _iter_cached_evals(
    conn: sqlite3.Connection,
    *,
    min_depth: int,
    max_rows: int | None,
    seed: int = _CACHE_SHUFFLE_SEED,
) -> Iterator[dict]:
    # Seed-stable shuffle: ordering by depth would bias first-N consumers
    # toward the deepest-analyzed (best-known theory) positions.
    conn.create_function(
        "eval_shuffle_key",
        1,
        lambda fen: hashlib.sha256(f"{seed}:{fen}".encode("utf-8")).hexdigest(),
    )
    query = (
        "SELECT fen, best_move, pv_line, depth, knodes, cp, mate "
        "FROM evals WHERE depth >= ? "
        "ORDER BY eval_shuffle_key(fen)"
    )
    params: list[int] = [min_depth]
    if max_rows is not None:
        query += " LIMIT ?"
        params.append(max_rows)

    for row in conn.execute(query, params):
        fen = row[0]
        try:
            pv_line = normalize_pv_line(fen, row[2])
        except (ValueError, TypeError):
            continue
        yield {
            "fen": fen,
            "best_move": pv_line.split()[0],
            "pv_line": pv_line,
            "depth": row[3],
            "knodes": row[4],
            "cp": row[5],
            "mate": row[6],
            "eval_perspective": EVAL_PERSPECTIVE,
        }


def partition_evals(evals: Iterator[dict]) -> dict[str, list[dict]]:
    """Partition filtered eval rows into task-specific buckets."""
    partitions: dict[str, list[dict]] = {
        "mate": [],
        "high_eval": [],
        "balanced": [],
        "endgame": [],
        "best_move": [],
    }

    for row in evals:
        board = board_from_raw(row)
        if board is None:
            continue
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


def _init_dedup_db(db_path: Path) -> sqlite3.Connection:
    """Create or open the dedup SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS evals (
            fen TEXT PRIMARY KEY,
            best_move TEXT,
            pv_line TEXT,
            depth INTEGER,
            knodes INTEGER,
            cp INTEGER,
            mate INTEGER
        )
        """
    )
    conn.commit()
    return conn


def _flush_batch(conn: sqlite3.Connection, batch: list[tuple]) -> None:
    """Insert or replace rows, keeping highest depth then highest knodes."""
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


def _coerce_int(value: object, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _default_dedup_db_path() -> Path:
    settings = SftDataSettings.from_env(Path.cwd())
    return settings.annotations_dir / "evals_dedup.db"


def _default_dataset_loader(*args, **kwargs):
    os.environ.setdefault("HF_HOME", DEFAULT_HF_CACHE_DIR)
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


__all__ = [
    "EVAL_PERSPECTIVE",
    "_flush_batch",
    "_init_dedup_db",
    "normalize_pv_line",
    "partition_evals",
    "parse_lichess_eval_row",
    "stream_evals",
]
