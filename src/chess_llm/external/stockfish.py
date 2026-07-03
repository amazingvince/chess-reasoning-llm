"""Stockfish process helpers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import chess
import chess.engine

logger = logging.getLogger(__name__)
EngineFactory = Callable[[str], Any]
EngineConfigurer = Callable[..., None]


@dataclass(frozen=True)
class StockfishEngineConfig:
    """Runtime configuration for a Stockfish UCI process."""

    path: str | Path
    threads: int = 1
    hash_mb: int = 256
    syzygy_path: str | Path | None = None
    limit_strength_elo: int | None = None


def open_stockfish(
    config: StockfishEngineConfig,
    *,
    engine_factory: EngineFactory | None = None,
    configure: EngineConfigurer | None = None,
) -> chess.engine.SimpleEngine:
    """Open and configure a Stockfish UCI engine."""
    engine_path = Path(config.path)
    if not engine_path.exists():
        raise ValueError(f"Stockfish binary not found: {engine_path}")

    factory = engine_factory or chess.engine.SimpleEngine.popen_uci
    configure_fn = configure or configure_stockfish_engine
    engine = factory(str(engine_path))
    try:
        configure_fn(
            engine,
            threads=config.threads,
            hash_mb=config.hash_mb,
            syzygy_path=config.syzygy_path,
            limit_strength_elo=config.limit_strength_elo,
        )
    except Exception:
        engine.quit()
        raise
    return engine


class StockfishWrapper:
    """Thin wrapper around python-chess's synchronous UCI engine interface."""

    def __init__(
        self,
        path: str | Path,
        depth: int = 15,
        threads: int = 1,
        hash_mb: int = 256,
        syzygy_path: str | Path | None = None,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        self.path = str(path)
        self.default_depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self.syzygy_path = syzygy_path
        self.engine_factory = engine_factory
        self._engine: Any | None = None

    def open(self) -> "StockfishWrapper":
        factory = self.engine_factory or chess.engine.SimpleEngine.popen_uci
        self._engine = factory(self.path)
        try:
            configure_stockfish_engine(
                self._engine,
                threads=self.threads,
                hash_mb=self.hash_mb,
                syzygy_path=self.syzygy_path,
            )
        except Exception:
            self.close()
            raise
        return self

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def evaluate(self, board: chess.Board, depth: int | None = None) -> dict:
        """Evaluate a board and return White-perspective cp/mate plus PV."""
        if self._engine is None:
            raise RuntimeError("Engine not opened; use as context manager")
        if not board.is_valid():
            return {"cp": None, "mate": None, "best_move": None, "pv_line": ""}

        search_depth = self.default_depth if depth is None else depth
        info = self._engine.analyse(
            board,
            chess.engine.Limit(depth=search_depth),
        )
        score = info["score"].white()
        cp = score.score() if not score.is_mate() else None
        mate = score.mate() if score.is_mate() else None
        pv = info.get("pv", [])

        return {
            "cp": cp,
            "mate": mate,
            "best_move": pv[0].uci() if pv else None,
            "pv_line": " ".join(move.uci() for move in pv),
        }

    def batch_evaluate(
        self,
        fens: list[str],
        depth: int | None = None,
    ) -> Iterator[dict]:
        """Evaluate multiple FEN strings sequentially."""
        for fen in fens:
            try:
                board = _board_from_fen(fen)
                if board is None:
                    raise ValueError("Invalid board state")
                yield self.evaluate(board, depth)
            except (ValueError, TypeError, chess.engine.EngineTerminatedError) as exc:
                logger.warning("Failed to evaluate %s: %s", fen, exc)
                yield {"cp": None, "mate": None, "best_move": None, "pv_line": ""}

    def __enter__(self) -> "StockfishWrapper":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()


def configure_stockfish_engine(
    engine: Any,
    *,
    threads: int = 1,
    hash_mb: int = 256,
    syzygy_path: str | Path | None = None,
    limit_strength_elo: int | None = None,
) -> None:
    """Apply common UCI options to a Stockfish-like engine.

    ``limit_strength_elo`` enables ``UCI_LimitStrength`` at the requested
    ``UCI_Elo``, clamped to the engine-reported option range; ``None`` leaves
    strength limiting untouched.
    """
    options: dict[str, bool | int | str] = {
        "Threads": int(threads),
        "Hash": int(hash_mb),
    }
    if syzygy_path is not None:
        options["SyzygyPath"] = str(syzygy_path)
    if limit_strength_elo is not None:
        options["UCI_LimitStrength"] = True
        options["UCI_Elo"] = clamp_uci_elo(engine, int(limit_strength_elo))
    configure = getattr(engine, "configure", None)
    if configure is not None:
        configure(options)


def clamp_uci_elo(engine: Any, elo: int) -> int:
    """Clamp a requested Elo to the engine-reported ``UCI_Elo`` range."""
    engine_options = getattr(engine, "options", None)
    option = None
    if engine_options is not None:
        try:
            option = engine_options.get("UCI_Elo")
        except (AttributeError, TypeError):
            option = None
    minimum = getattr(option, "min", None)
    maximum = getattr(option, "max", None)
    if minimum is not None:
        elo = max(int(minimum), elo)
    if maximum is not None:
        elo = min(int(maximum), elo)
    return elo


def stockfish_engine_name(engine: Any) -> str | None:
    """Return the engine-reported name when python-chess exposes one."""
    engine_id = getattr(engine, "id", None)
    if isinstance(engine_id, dict):
        name = engine_id.get("name")
        return str(name) if name else None
    return None


def _board_from_fen(fen: str) -> chess.Board | None:
    """Parse a standard or Chess960 FEN and reject invalid board states."""
    for chess960 in (False, True):
        try:
            board = chess.Board(fen, chess960=chess960)
        except (TypeError, ValueError):
            continue
        if board.is_valid():
            return board
    return None
