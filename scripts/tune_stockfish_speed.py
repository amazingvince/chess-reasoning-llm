#!/usr/bin/env python3
"""Benchmark Stockfish MultiPV throughput across worker/thread settings."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil
import sys
import time
from typing import NamedTuple

import chess
import chess.engine


class ProbeSetting(NamedTuple):
    workers: int
    threads_per_worker: int

    def as_dict(self) -> dict[str, int]:
        return {
            "workers": self.workers,
            "threads_per_worker": self.threads_per_worker,
        }


def _parse_csv_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one integer")
    if any(item <= 0 for item in parsed):
        raise ValueError("all integers must be > 0")
    return parsed


def build_probe_matrix(worker_counts: str, threads_per_worker: str) -> list[ProbeSetting]:
    """Return worker/thread combinations in deterministic nested-loop order."""
    return [
        ProbeSetting(workers=workers, threads_per_worker=threads)
        for workers in _parse_csv_ints(worker_counts)
        for threads in _parse_csv_ints(threads_per_worker)
    ]


def _extract_fen(row: dict[str, object]) -> str | None:
    fen = row.get("fen")
    if isinstance(fen, str) and fen.strip():
        return fen.strip()
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("fen", "start_fen", "position_fen"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def load_fens_from_jsonl(path: str | Path, limit: int | None = None) -> list[str]:
    """Load unique FENs from benchmark/data JSONL, preserving first-seen order."""
    fens: list[str] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                continue
            fen = _extract_fen(row)
            if fen is None or fen in seen:
                continue
            seen.add(fen)
            fens.append(fen)
            if limit is not None and len(fens) >= limit:
                break
    return fens


def _board_from_fen(fen: str) -> chess.Board | None:
    for chess960 in (False, True):
        try:
            board = chess.Board(fen, chess960=chess960)
        except (TypeError, ValueError):
            continue
        if board.is_valid():
            return board
    return None


def _resolve_stockfish_path(stockfish_path: str | Path) -> str:
    value = str(stockfish_path)
    direct_path = Path(value)
    if direct_path.exists():
        return str(direct_path)
    resolved = shutil.which(value)
    if resolved:
        return resolved
    raise FileNotFoundError(f"Stockfish binary not found: {stockfish_path}")


def _split_fens(fens: list[str], workers: int) -> list[list[str]]:
    actual_workers = max(1, min(int(workers), len(fens) or 1))
    chunks: list[list[str]] = [[] for _ in range(actual_workers)]
    for index, fen in enumerate(fens):
        chunks[index % actual_workers].append(fen)
    return [chunk for chunk in chunks if chunk]


def _analyse_chunk(
    fens: list[str],
    *,
    stockfish_path: str,
    depth: int,
    multipv: int,
    threads_per_worker: int,
    hash_mb: int,
) -> dict[str, int | float]:
    started = time.perf_counter()
    ok = 0
    errors = 0
    pv_rows = 0
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        engine.configure(
            {
                "Threads": int(threads_per_worker),
                "Hash": int(hash_mb),
            }
        )
        limit = chess.engine.Limit(depth=int(depth))
        for fen in fens:
            board = _board_from_fen(fen)
            if board is None:
                errors += 1
                continue
            try:
                info = engine.analyse(board, limit, multipv=int(multipv))
            except (chess.engine.EngineError, chess.engine.EngineTerminatedError, TimeoutError):
                errors += 1
                continue
            rows = info if isinstance(info, list) else [info]
            ok += 1
            pv_rows += len(rows)
    finally:
        engine.quit()
    return {
        "ok": ok,
        "errors": errors,
        "pv_rows": pv_rows,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def run_stockfish_probe(
    fens: list[str],
    *,
    stockfish_path: str | Path,
    setting: ProbeSetting,
    depth: int,
    multipv: int,
    hash_mb: int,
) -> dict[str, object]:
    """Run one Stockfish MultiPV throughput probe."""
    resolved_path = _resolve_stockfish_path(stockfish_path)
    chunks = _split_fens(fens, setting.workers)
    started = time.perf_counter()
    totals = {"ok": 0, "errors": 0, "pv_rows": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks) or 1) as pool:
        futures = [
            pool.submit(
                _analyse_chunk,
                chunk,
                stockfish_path=resolved_path,
                depth=depth,
                multipv=multipv,
                threads_per_worker=setting.threads_per_worker,
                hash_mb=hash_mb,
            )
            for chunk in chunks
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            totals["ok"] += int(result["ok"])
            totals["errors"] += int(result["errors"])
            totals["pv_rows"] += int(result["pv_rows"])
    elapsed = time.perf_counter() - started
    analysed = totals["ok"]
    return {
        **setting.as_dict(),
        "actual_workers": len(chunks),
        "depth": depth,
        "multipv": multipv,
        "hash_mb_per_worker": hash_mb,
        "total_hash_mb": hash_mb * max(len(chunks), 1),
        "fen_count": len(fens),
        **totals,
        "elapsed_seconds": round(elapsed, 3),
        "positions_per_second": round(analysed / elapsed, 3) if elapsed > 0 else 0.0,
        "pv_rows_per_second": round(totals["pv_rows"] / elapsed, 3) if elapsed > 0 else 0.0,
    }


def _write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Stockfish MultiPV Speed Probe",
        "",
        "| workers | threads | depth | multipv | fen | ok | errors | elapsed_s | pos/s | pv rows/s | hash_mb |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {workers} | {threads_per_worker} | {depth} | {multipv} | {fen_count} | "
            "{ok} | {errors} | {elapsed_seconds} | {positions_per_second} | "
            "{pv_rows_per_second} | {total_hash_mb} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune Stockfish MultiPV throughput")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--fen-file", type=Path, help="JSONL file containing top-level or metadata.fen fields")
    input_group.add_argument("--benchmark-dir", type=Path, help="Benchmark directory containing split JSONL files")
    parser.add_argument("--split", default="planning", help="Split name when --benchmark-dir is used")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--multipv", type=int, default=5)
    parser.add_argument("--worker-counts", default="1,4,8,16")
    parser.add_argument("--threads-per-worker", default="1,2,4")
    parser.add_argument("--hash-mb", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    fen_file = args.fen_file
    if fen_file is None:
        fen_file = args.benchmark_dir / f"{args.split}.jsonl"
    fens = load_fens_from_jsonl(fen_file, limit=args.limit)
    if not fens:
        print(f"No FENs found in {fen_file}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    try:
        for setting in build_probe_matrix(args.worker_counts, args.threads_per_worker):
            row = run_stockfish_probe(
                fens,
                stockfish_path=args.stockfish_path,
                setting=setting,
                depth=args.depth,
                multipv=args.multipv,
                hash_mb=args.hash_mb,
            )
            rows.append(row)
            print(
                "workers={workers} threads={threads_per_worker} "
                "elapsed={elapsed_seconds}s pos/s={positions_per_second}".format(**row)
            )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    summary_path = args.output_dir / "stockfish_speed_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_markdown(rows, args.output_dir / "stockfish_speed_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
