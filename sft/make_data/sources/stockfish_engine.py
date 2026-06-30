"""Compatibility wrapper for package-owned Stockfish helpers."""

from __future__ import annotations

try:
    from config.settings import STOCKFISH_PATH as _STOCKFISH_PATH
except ModuleNotFoundError:
    from sft.make_data.config.settings import STOCKFISH_PATH as _STOCKFISH_PATH

from chess_llm.external.stockfish import StockfishWrapper as _PackageStockfishWrapper


class StockfishWrapper(_PackageStockfishWrapper):
    """Legacy default-path wrapper around the package Stockfish wrapper."""

    def __init__(
        self,
        path: str | None = None,
        depth: int = 15,
        threads: int = 1,
        hash_mb: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(
            path=path or _STOCKFISH_PATH,
            depth=depth,
            threads=threads,
            hash_mb=hash_mb,
            **kwargs,
        )


__all__ = ["StockfishWrapper"]
