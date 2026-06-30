"""Shared chess and Stockfish helpers for the UI backend."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from pathlib import Path
from typing import Any

import chess
import chess.engine

from chess_llm.artifacts.schemas import JudgmentArtifact, PromptArtifact, RolloutArtifact
from chess_llm.autodata.judge import judge_rollout
from chess_llm.autodata.stockfish_judge import judge_rollout_with_stockfish
from chess_llm.core.opening_books import discover_opening_books, get_weighted_book_moves
from chess_llm.external.stockfish import (
    StockfishEngineConfig,
    configure_stockfish_engine,
    open_stockfish,
    stockfish_engine_name,
)

from chess_llm_ui.config import BackendSettings


@dataclass
class ToolService:
    """Own chess helper APIs and an optional serialized Stockfish engine."""

    settings: BackendSettings
    engine: Any | None = None
    _engine_lock: threading.RLock = field(default_factory=threading.RLock)
    _owns_engine: bool = False
    _last_error: str | None = None
    _engine_name: str | None = None

    def __post_init__(self) -> None:
        if self.engine is None:
            return
        self._engine_name = stockfish_engine_name(self.engine)
        try:
            configure_stockfish_engine(
                self.engine,
                threads=self.settings.stockfish_threads,
                hash_mb=self.settings.stockfish_hash_mb,
            )
        except Exception as exc:
            self._last_error = str(exc)
            quit_engine = getattr(self.engine, "quit", None)
            if quit_engine is not None:
                quit_engine()
            self.engine = None

    def status(self) -> dict:
        enabled = self._stockfish_enabled()
        available = self._stockfish_available_without_opening()
        return {
            "stockfish": {
                "enabled": enabled,
                "available": available,
                "path": str(self.settings.stockfish_path) if self.settings.stockfish_path else None,
                "name": self._engine_name,
                "depth": self.settings.stockfish_depth,
                "threads": self.settings.stockfish_threads,
                "hash_mb": self.settings.stockfish_hash_mb,
                "error": self._last_error,
            },
            "capabilities": {
                "legal_moves": True,
                "opening_book_moves": True,
                "stockfish_analysis": available,
            },
            "batch_limit": self.settings.tool_batch_limit,
        }

    def close(self) -> None:
        with self._engine_lock:
            if self.engine is not None and self._owns_engine:
                self.engine.quit()
            self.engine = None
            self._owns_engine = False

    def judge_rollout(
        self,
        prompt: PromptArtifact,
        rollout: RolloutArtifact,
        *,
        metadata: dict[str, Any] | None = None,
        depth: int | None = None,
    ) -> JudgmentArtifact:
        resolved_metadata = dict(metadata or {})
        engine = self._get_engine()
        if engine is None:
            return judge_rollout(
                prompt,
                rollout,
                metadata={
                    **resolved_metadata,
                    "judge": "legality",
                    "stockfish_available": False,
                    "stockfish_error": self._last_error,
                },
            )

        search_depth = self.settings.stockfish_depth if depth is None else int(depth)
        try:
            with self._engine_lock:
                judgment = judge_rollout_with_stockfish(
                    prompt,
                    rollout,
                    engine,
                    depth=search_depth,
                    metadata={**resolved_metadata, "stockfish_available": True},
                )
        except Exception as exc:
            self._last_error = str(exc)
            return judge_rollout(
                prompt,
                rollout,
                metadata={
                    **resolved_metadata,
                    "judge": "legality",
                    "stockfish_available": False,
                    "stockfish_error": str(exc),
                },
            )

        if judgment.metadata.get("stockfish_scored") is False:
            return self._legality_fallback_after_stockfish_error(
                prompt,
                rollout,
                resolved_metadata,
                judgment,
            )
        return judgment

    def analyze_position(
        self,
        *,
        fen: str,
        book_id: str | None = None,
        include_stockfish: bool = False,
        depth: int | None = None,
    ) -> dict:
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError("invalid board state")

        legal_moves = [
            _move_metadata(board, move)
            for move in sorted(board.legal_moves, key=lambda candidate: candidate.uci())
        ]
        book_moves = self._book_moves(board, book_id)
        stockfish = None
        if include_stockfish:
            stockfish = self._stockfish_analysis(board, depth=depth)
        return {
            "fen": board.fen(),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "legal_move_count": len(legal_moves),
            "legal_moves": legal_moves,
            "book_moves": book_moves,
            "stockfish": stockfish,
        }

    def _book_moves(self, board: chess.Board, book_id: str | None) -> list[dict]:
        if not book_id:
            return []
        matching = [
            book for book in discover_opening_books(self.settings.book_root)
            if book.book_id == book_id
        ]
        if not matching:
            return []

        try:
            weighted_moves = get_weighted_book_moves(matching[0].path, board)
        except Exception as exc:
            self._last_error = str(exc)
            return []

        moves: list[dict] = []
        for move_uci, weight in weighted_moves:
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                continue
            if move not in board.legal_moves:
                continue
            metadata = _move_metadata(board, move)
            metadata["weight"] = int(weight)
            moves.append(metadata)
        return moves

    def _stockfish_analysis(self, board: chess.Board, *, depth: int | None = None) -> dict:
        engine = self._get_engine()
        search_depth = self.settings.stockfish_depth if depth is None else int(depth)
        if engine is None:
            return {"available": False, "error": self._last_error, "depth": search_depth}
        try:
            with self._engine_lock:
                info = engine.analyse(board, chess.engine.Limit(depth=search_depth))
        except Exception as exc:
            self._last_error = str(exc)
            return {"available": False, "error": str(exc), "depth": search_depth}

        score = info.get("score")
        cp = None
        mate = None
        if score is not None:
            white_score = score.white() if hasattr(score, "white") else score
            if hasattr(white_score, "is_mate") and white_score.is_mate():
                mate = white_score.mate()
            elif hasattr(white_score, "score"):
                cp = white_score.score()
        pv = info.get("pv") or []
        return {
            "available": True,
            "best_move": pv[0].uci() if pv else None,
            "cp": cp,
            "mate": mate,
            "pv_line": " ".join(move.uci() for move in pv),
            "depth": search_depth,
        }

    def _get_engine(self) -> Any | None:
        with self._engine_lock:
            if self.engine is not None:
                return self.engine
            path = self.settings.stockfish_path
            if path is None:
                self._last_error = "CHESS_UI_STOCKFISH_PATH is not configured"
                return None
            path = Path(path)
            if not path.exists():
                self._last_error = f"Stockfish binary not found: {path}"
                return None
            try:
                self.engine = open_stockfish(
                    StockfishEngineConfig(
                        path=path,
                        threads=self.settings.stockfish_threads,
                        hash_mb=self.settings.stockfish_hash_mb,
                    )
                )
            except Exception as exc:
                self._last_error = str(exc)
                self.engine = None
                return None
            self._owns_engine = True
            self._engine_name = stockfish_engine_name(self.engine)
            self._last_error = None
            return self.engine

    def _stockfish_enabled(self) -> bool:
        return self.engine is not None or self.settings.stockfish_path is not None

    def _stockfish_available_without_opening(self) -> bool:
        if self.engine is not None:
            return self._last_error is None
        path = self.settings.stockfish_path
        if path is None:
            return False
        if not Path(path).exists():
            self._last_error = f"Stockfish binary not found: {path}"
            return False
        return True

    def _legality_fallback_after_stockfish_error(
        self,
        prompt: PromptArtifact,
        rollout: RolloutArtifact,
        metadata: dict[str, Any],
        stockfish_judgment: JudgmentArtifact,
    ) -> JudgmentArtifact:
        fallback = judge_rollout(
            prompt,
            rollout,
            metadata={
                **metadata,
                "judge": "legality",
                "stockfish_available": True,
                "stockfish_error": stockfish_judgment.metadata.get("stockfish_error"),
            },
        )
        if fallback.feedback and stockfish_judgment.metadata.get("stockfish_error"):
            fallback.feedback = (
                f"{fallback.feedback} Stockfish scoring failed: "
                f"{stockfish_judgment.metadata['stockfish_error']}"
            )
        return fallback


def _move_metadata(board: chess.Board, move: chess.Move) -> dict:
    return {
        "uci": move.uci(),
        "san": board.san(move),
        "capture": board.is_capture(move),
        "check": board.gives_check(move),
        "promotion": chess.piece_name(move.promotion) if move.promotion else None,
    }
