"""Stockfish process helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess.engine


@dataclass(frozen=True)
class StockfishEngineConfig:
    """Runtime configuration for a Stockfish UCI process."""

    path: str | Path
    threads: int = 1
    hash_mb: int = 256
    syzygy_path: str | Path | None = None


def open_stockfish(config: StockfishEngineConfig) -> chess.engine.SimpleEngine:
    """Open and configure a Stockfish UCI engine."""
    engine_path = Path(config.path)
    if not engine_path.exists():
        raise ValueError(f"Stockfish binary not found: {engine_path}")

    engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    try:
        configure_stockfish_engine(
            engine,
            threads=config.threads,
            hash_mb=config.hash_mb,
            syzygy_path=config.syzygy_path,
        )
    except Exception:
        engine.quit()
        raise
    return engine


def configure_stockfish_engine(
    engine: Any,
    *,
    threads: int = 1,
    hash_mb: int = 256,
    syzygy_path: str | Path | None = None,
) -> None:
    """Apply common UCI options to a Stockfish-like engine."""
    options: dict[str, int | str] = {
        "Threads": int(threads),
        "Hash": int(hash_mb),
    }
    if syzygy_path is not None:
        options["SyzygyPath"] = str(syzygy_path)
    configure = getattr(engine, "configure", None)
    if configure is not None:
        configure(options)


def stockfish_engine_name(engine: Any) -> str | None:
    """Return the engine-reported name when python-chess exposes one."""
    engine_id = getattr(engine, "id", None)
    if isinstance(engine_id, dict):
        name = engine_id.get("name")
        return str(name) if name else None
    return None
