"""Source 7: Stockfish UCI Engine Wrapper.

Provides batch evaluation of chess positions via Stockfish.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Iterator

import chess
import chess.engine

from config.settings import STOCKFISH_PATH

logger = logging.getLogger(__name__)


class StockfishWrapper:
    """Thin wrapper around python-chess's UCI engine interface.

    Usage::

        with StockfishWrapper() as sf:
            result = sf.evaluate(board)
            # result = {cp, mate, best_move, pv_line}

    Parameters
    ----------
    path : str
        Path to the Stockfish binary.
    depth : int
        Default search depth.
    threads : int
        Number of search threads.
    hash_mb : int
        Hash table size in MB.
    """

    def __init__(
        self,
        path: str | None = None,
        depth: int = 15,
        threads: int = 1,
        hash_mb: int = 256,
    ) -> None:
        self.path = path or STOCKFISH_PATH
        self.default_depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self._engine: chess.engine.SimpleEngine | None = None

    def open(self) -> StockfishWrapper:
        self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        self._engine.configure({
            "Threads": self.threads,
            "Hash": self.hash_mb,
        })
        return self

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def evaluate(self, board: chess.Board, depth: int | None = None) -> dict:
        """Evaluate a single position.

        Returns ``{cp, mate, best_move, pv_line}``.
        """
        if self._engine is None:
            raise RuntimeError("Engine not opened; use as context manager")

        d = depth or self.default_depth
        info = self._engine.analyse(board, chess.engine.Limit(depth=d))

        score = info["score"].relative
        cp = score.score() if not score.is_mate() else None
        mate = score.mate() if score.is_mate() else None

        pv = info.get("pv", [])
        best_move = pv[0].uci() if pv else None
        pv_line = " ".join(m.uci() for m in pv)

        return {
            "cp": cp,
            "mate": mate,
            "best_move": best_move,
            "pv_line": pv_line,
        }

    def batch_evaluate(
        self,
        fens: list[str],
        depth: int | None = None,
    ) -> Iterator[dict]:
        """Evaluate multiple FENs sequentially.

        Yields one result dict per FEN.
        """
        for fen in fens:
            try:
                board = chess.Board(fen)
                yield self.evaluate(board, depth)
            except (ValueError, chess.engine.EngineTerminatedError) as exc:
                logger.warning("Failed to evaluate %s: %s", fen, exc)
                yield {"cp": None, "mate": None, "best_move": None, "pv_line": ""}

    def __enter__(self) -> StockfishWrapper:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()
