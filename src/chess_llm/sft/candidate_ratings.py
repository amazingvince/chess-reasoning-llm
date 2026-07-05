"""Build true-MultiPV candidate-rating rows for the R4 SFT task."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from chess_llm.external.multipv import SqliteMultipvCache, analyze_multipv
from chess_llm.external.stockfish import StockfishEngineConfig, open_stockfish
from chess_llm.sft.context import raw_is_chess960
from chess_llm.sft.settings import DEFAULT_STOCKFISH_PATH, SftDataSettings

logger = logging.getLogger(__name__)

_SETTINGS = SftDataSettings.from_env(Path.cwd())
DEFAULT_OUTPUT_PATH = _SETTINGS.annotations_dir / "candidate_ratings.jsonl"
DEFAULT_CACHE_PATH = _SETTINGS.annotations_dir / "multipv.sqlite"
DEFAULT_WORKERS = max(1, min(os.cpu_count() or 1, 8))
EngineOpener = Callable[[StockfishEngineConfig], Any]


@dataclass(frozen=True)
class CandidateRatingBuildResult:
    """Rows and counters from a candidate-rating build."""

    rows: list[dict[str, Any]]
    input_count: int
    row_count: int
    skipped_count: int


def build_candidate_rating_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    engine: Any,
    cache_path: str | Path,
    depth: int,
    multipv: int = 5,
    pv_len: int = 8,
    max_rows: int = 0,
    force_chess960: bool = False,
    engine_config: Mapping[str, Any] | None = None,
) -> CandidateRatingBuildResult:
    """Analyze input FEN rows and return rows with exactly five candidate ratings."""
    cache = SqliteMultipvCache(cache_path)
    output_rows: list[dict[str, Any]] = []
    input_count = 0
    skipped_count = 0

    for row in rows:
        if max_rows > 0 and input_count >= max_rows:
            break
        input_count += 1
        result_row = _build_candidate_rating_row(
            row,
            engine=engine,
            cache=cache,
            depth=depth,
            multipv=multipv,
            pv_len=pv_len,
            force_chess960=force_chess960,
            engine_config=engine_config,
        )
        if result_row is None:
            skipped_count += 1
            continue
        output_rows.append(result_row)

    return CandidateRatingBuildResult(
        rows=output_rows,
        input_count=input_count,
        row_count=len(output_rows),
        skipped_count=skipped_count,
    )


def build_candidate_rating_rows_with_workers(
    rows: Iterable[Mapping[str, Any]],
    *,
    stockfish_path: str | Path,
    cache_path: str | Path,
    depth: int,
    multipv: int = 5,
    pv_len: int = 8,
    max_rows: int = 0,
    force_chess960: bool = False,
    workers: int = DEFAULT_WORKERS,
    threads_per_worker: int = 1,
    hash_mb: int = 256,
    engine_config: Mapping[str, Any] | None = None,
    engine_opener: EngineOpener = open_stockfish,
) -> CandidateRatingBuildResult:
    """Analyze rows with one Stockfish process per worker and stable output order."""
    indexed_rows: list[tuple[int, Mapping[str, Any]]] = []
    input_count = 0
    for row in rows:
        if max_rows > 0 and input_count >= max_rows:
            break
        indexed_rows.append((input_count, row))
        input_count += 1

    if not indexed_rows:
        return CandidateRatingBuildResult(
            rows=[],
            input_count=0,
            row_count=0,
            skipped_count=0,
        )

    worker_count = max(1, min(int(workers), len(indexed_rows)))
    chunks: list[list[tuple[int, Mapping[str, Any]]]] = [
        [] for _ in range(worker_count)
    ]
    for offset, item in enumerate(indexed_rows):
        chunks[offset % worker_count].append(item)

    output_items: list[tuple[int, dict[str, Any]]] = []
    skipped_count = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _candidate_rating_worker,
                chunk,
                stockfish_path=stockfish_path,
                cache_path=cache_path,
                depth=depth,
                multipv=multipv,
                pv_len=pv_len,
                force_chess960=force_chess960,
                threads_per_worker=threads_per_worker,
                hash_mb=hash_mb,
                engine_config=engine_config,
                engine_opener=engine_opener,
            )
            for chunk in chunks
            if chunk
        ]
        for future in as_completed(futures):
            worker_rows, worker_skipped = future.result()
            output_items.extend(worker_rows)
            skipped_count += worker_skipped

    output_rows = [
        row for _index, row in sorted(output_items, key=lambda item: item[0])
    ]

    return CandidateRatingBuildResult(
        rows=output_rows,
        input_count=input_count,
        row_count=len(output_rows),
        skipped_count=skipped_count,
    )


def write_candidate_rating_jsonl(
    rows: Iterable[Mapping[str, Any]],
    *,
    output_path: str | Path,
    engine: Any | None,
    cache_path: str | Path,
    depth: int,
    multipv: int = 5,
    pv_len: int = 8,
    max_rows: int = 0,
    force_chess960: bool = False,
    engine_config: Mapping[str, Any] | None = None,
    stockfish_path: str | Path | None = None,
    workers: int = 1,
    threads_per_worker: int = 1,
    hash_mb: int = 256,
    engine_opener: EngineOpener = open_stockfish,
) -> CandidateRatingBuildResult:
    """Build candidate-rating rows and write them as JSONL."""
    if engine is None:
        if stockfish_path is None:
            raise ValueError("stockfish_path is required when engine is None")
        result = build_candidate_rating_rows_with_workers(
            rows,
            stockfish_path=stockfish_path,
            cache_path=cache_path,
            depth=depth,
            multipv=multipv,
            pv_len=pv_len,
            max_rows=max_rows,
            force_chess960=force_chess960,
            workers=workers,
            threads_per_worker=threads_per_worker,
            hash_mb=hash_mb,
            engine_config=engine_config,
            engine_opener=engine_opener,
        )
    else:
        result = build_candidate_rating_rows(
            rows,
            engine=engine,
            cache_path=cache_path,
            depth=depth,
            multipv=multipv,
            pv_len=pv_len,
            max_rows=max_rows,
            force_chess960=force_chess960,
            engine_config=engine_config,
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in result.rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result


def load_candidate_rating_evals(path: str | Path) -> list[dict[str, Any]]:
    """Load previously built candidate-rating rows from JSONL."""
    candidate_path = Path(path)
    if not candidate_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with candidate_path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(row, dict)
                and row.get("fen")
                and isinstance(row.get("candidate_ratings"), list)
            ):
                rows.append(row)
    return rows


def load_input_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load input JSONL rows that contain at least a FEN."""
    input_path = Path(path)
    rows: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("fen"):
                rows.append(row)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build true Stockfish MultiPV rows for 7.8_candidate_ratings."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--stockfish-path", default=DEFAULT_STOCKFISH_PATH)
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--multipv", type=int, default=5)
    parser.add_argument("--pv-len", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=256)
    parser.add_argument("--chess960", action="store_true", help="Force Chess960 parsing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.multipv < 5:
        print("--multipv must be at least 5 for candidate ratings.")
        return 2
    if args.depth <= 0:
        print("--depth must be positive.")
        return 2
    if args.max_rows < 0:
        print("--max-rows must be greater than or equal to 0.")
        return 2
    if args.workers <= 0:
        print("--workers must be positive.")
        return 2
    if args.threads <= 0:
        print("--threads must be positive.")
        return 2
    if not args.input_jsonl.exists():
        print(f"Input JSONL does not exist: {args.input_jsonl}")
        return 1

    stockfish_path = _resolve_stockfish_path(args.stockfish_path)
    if stockfish_path is None:
        print(f"Stockfish binary not found: {args.stockfish_path}")
        return 1

    rows = load_input_rows(args.input_jsonl)
    result = write_candidate_rating_jsonl(
        rows,
        output_path=args.output_jsonl,
        engine=None,
        stockfish_path=stockfish_path,
        cache_path=args.cache,
        depth=args.depth,
        multipv=args.multipv,
        pv_len=args.pv_len,
        max_rows=args.max_rows,
        force_chess960=args.chess960,
        workers=args.workers,
        threads_per_worker=args.threads,
        hash_mb=args.hash_mb,
        engine_config={
            "threads": args.threads,
            "hash_mb": args.hash_mb,
            "stockfish_path": str(stockfish_path),
        },
    )
    print(
        "Candidate ratings: "
        f"wrote {result.row_count} rows to {args.output_jsonl} "
        f"({result.skipped_count} skipped / {result.input_count} input)."
    )
    return 0


def _candidate_rating_payload(move: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "uci": move.uci,
        "rank": move.rank,
        "expectation": move.expectation,
        "pv_line": " ".join(move.pv),
    }
    if move.cp is not None:
        payload["cp"] = move.cp
    if move.mate is not None:
        payload["mate"] = move.mate
    return payload


def _build_candidate_rating_row(
    row: Mapping[str, Any],
    *,
    engine: Any,
    cache: SqliteMultipvCache,
    depth: int,
    multipv: int,
    pv_len: int,
    force_chess960: bool,
    engine_config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    fen = str(row.get("fen", "") or "")
    if not fen:
        return None
    chess960 = bool(force_chess960 or raw_is_chess960(row))
    try:
        board = chess.Board(fen, chess960=chess960)
    except (TypeError, ValueError):
        return None
    if not board.is_valid():
        return None

    try:
        analysis = analyze_multipv(
            engine,
            fen,
            depth=depth,
            k=multipv,
            pv_len=pv_len,
            cache=cache,
            chess960=chess960,
            engine_config=engine_config,
        )
    except Exception as exc:
        logger.warning("Skipping %s after MultiPV failure: %s", fen, exc)
        return None
    if len(analysis.moves) < 5:
        return None

    candidate_ratings = [
        _candidate_rating_payload(move)
        for move in analysis.moves[:5]
    ]
    result_row = dict(row)
    result_row.update(
        {
            "fen": fen,
            "source": "stockfish_multipv",
            "best_move": candidate_ratings[0]["uci"],
            "candidate_ratings": candidate_ratings,
            "multipv_depth": depth,
            "multipv_k": multipv,
            "pv_len": pv_len,
            "is_chess960": chess960,
        }
    )
    return result_row


def _candidate_rating_worker(
    indexed_rows: list[tuple[int, Mapping[str, Any]]],
    *,
    stockfish_path: str | Path,
    cache_path: str | Path,
    depth: int,
    multipv: int,
    pv_len: int,
    force_chess960: bool,
    threads_per_worker: int,
    hash_mb: int,
    engine_config: Mapping[str, Any] | None,
    engine_opener: EngineOpener,
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    resolved_engine_config = dict(engine_config or {})
    resolved_engine_config.setdefault("threads", int(threads_per_worker))
    resolved_engine_config.setdefault("hash_mb", int(hash_mb))
    resolved_engine_config.setdefault("stockfish_path", str(stockfish_path))
    engine = engine_opener(
        StockfishEngineConfig(
            stockfish_path,
            threads=int(threads_per_worker),
            hash_mb=int(hash_mb),
        )
    )
    try:
        cache = SqliteMultipvCache(cache_path)
        output_rows: list[tuple[int, dict[str, Any]]] = []
        skipped_count = 0
        for index, row in indexed_rows:
            result_row = _build_candidate_rating_row(
                row,
                engine=engine,
                cache=cache,
                depth=depth,
                multipv=multipv,
                pv_len=pv_len,
                force_chess960=force_chess960,
                engine_config=resolved_engine_config,
            )
            if result_row is None:
                skipped_count += 1
            else:
                output_rows.append((index, result_row))
        return output_rows, skipped_count
    finally:
        engine.quit()


def _resolve_stockfish_path(value: str | Path) -> Path | None:
    path = Path(value)
    if path.exists():
        return path
    resolved = shutil.which(str(value))
    return Path(resolved) if resolved else None


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CandidateRatingBuildResult",
    "DEFAULT_CACHE_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_WORKERS",
    "build_arg_parser",
    "build_candidate_rating_rows",
    "build_candidate_rating_rows_with_workers",
    "load_candidate_rating_evals",
    "load_input_rows",
    "main",
    "write_candidate_rating_jsonl",
]
