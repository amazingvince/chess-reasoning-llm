"""Shared Stockfish MultiPV analysis and WPD diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Mapping

import chess
import chess.engine

from chess_llm.external.stockfish import stockfish_engine_name


_CACHE_VERSION = 1


@dataclass(frozen=True)
class MultipvMove:
    """One root move from a MultiPV analysis, scored from side-to-move POV."""

    uci: str
    rank: int
    cp: int | None
    mate: int | None
    expectation: float
    pv: list[str]


@dataclass(frozen=True)
class MultipvAnalysis:
    """Cached MultiPV result for a single root position."""

    fen: str
    chess960: bool
    depth: int
    k: int
    pv_len: int
    moves: list[MultipvMove]
    engine_metadata: dict[str, Any]


@dataclass(frozen=True)
class WpdDiagnostics:
    """Win-probability-difference reward diagnostics for one predicted move."""

    predicted_move: str
    best_move: str | None
    best_expectation: float | None
    predicted_expectation: float | None
    wpd: float | None
    reward: float
    reward_bucket: str
    multipv_hit: bool
    postmove_hit: bool
    cache_hit: bool

    def to_metric_dict(self) -> dict[str, float | bool | str | None]:
        return asdict(self)


class SqliteMultipvCache:
    """Tiny SQLite cache keyed by FEN, variant, depth, MultiPV, PV length, config."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.last_hit = False
        self._init_db()

    def get(
        self,
        *,
        fen: str,
        chess960: bool,
        depth: int,
        k: int,
        pv_len: int,
        engine_config: Mapping[str, Any] | None = None,
    ) -> MultipvAnalysis | None:
        key = self._key(
            fen=fen,
            chess960=chess960,
            depth=depth,
            k=k,
            pv_len=pv_len,
            engine_config=engine_config,
        )
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT payload FROM multipv_analyses WHERE cache_key = ?",
                (key,),
            ).fetchone()
        self.last_hit = row is not None
        if row is None:
            return None
        payload = json.loads(row[0])
        return _analysis_from_payload(payload)

    def put(
        self,
        analysis: MultipvAnalysis,
        *,
        engine_config: Mapping[str, Any] | None = None,
    ) -> None:
        key = self._key(
            fen=analysis.fen,
            chess960=analysis.chess960,
            depth=analysis.depth,
            k=analysis.k,
            pv_len=analysis.pv_len,
            engine_config=engine_config,
        )
        payload = _analysis_payload(analysis)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO multipv_analyses
                    (cache_key, fen, chess960, depth, multipv, pv_len,
                     engine_config, payload, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    analysis.fen,
                    int(analysis.chess960),
                    analysis.depth,
                    analysis.k,
                    analysis.pv_len,
                    _stable_json(engine_config or {}),
                    json.dumps(payload, sort_keys=True),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_scalar_evaluation(
        self,
        *,
        kind: str,
        fen: str,
        chess960: bool,
        depth: int,
        move_uci: str | None = None,
        pov: str = "side_to_move",
        engine_config: Mapping[str, Any] | None = None,
    ) -> tuple[bool, int | None]:
        """Return ``(hit, cp)`` for a cached scalar Stockfish evaluation."""
        key = self._scalar_key(
            kind=kind,
            fen=fen,
            chess960=chess960,
            depth=depth,
            move_uci=move_uci,
            pov=pov,
            engine_config=engine_config,
        )
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT cp FROM scalar_evaluations WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return False, None
        return True, int(row[0]) if row[0] is not None else None

    def put_scalar_evaluation(
        self,
        *,
        kind: str,
        fen: str,
        chess960: bool,
        depth: int,
        cp: int | None,
        move_uci: str | None = None,
        pov: str = "side_to_move",
        engine_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Cache one scalar Stockfish evaluation."""
        key = self._scalar_key(
            kind=kind,
            fen=fen,
            chess960=chess960,
            depth=depth,
            move_uci=move_uci,
            pov=pov,
            engine_config=engine_config,
        )
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scalar_evaluations
                    (cache_key, kind, fen, chess960, depth, move_uci, pov,
                     engine_config, cp, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    kind,
                    fen,
                    int(chess960),
                    int(depth),
                    move_uci,
                    pov,
                    _stable_json(engine_config or {}),
                    cp,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS multipv_analyses (
                    cache_key TEXT PRIMARY KEY,
                    fen TEXT NOT NULL,
                    chess960 INTEGER NOT NULL,
                    depth INTEGER NOT NULL,
                    multipv INTEGER NOT NULL,
                    pv_len INTEGER NOT NULL,
                    engine_config TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scalar_evaluations (
                    cache_key TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    fen TEXT NOT NULL,
                    chess960 INTEGER NOT NULL,
                    depth INTEGER NOT NULL,
                    move_uci TEXT,
                    pov TEXT NOT NULL,
                    engine_config TEXT NOT NULL,
                    cp INTEGER,
                    created_at_utc TEXT NOT NULL
                )
                """
            )

    def _key(
        self,
        *,
        fen: str,
        chess960: bool,
        depth: int,
        k: int,
        pv_len: int,
        engine_config: Mapping[str, Any] | None = None,
    ) -> str:
        payload = {
            "version": _CACHE_VERSION,
            "fen": fen,
            "chess960": bool(chess960),
            "depth": int(depth),
            "k": int(k),
            "pv_len": int(pv_len),
            "engine_config": engine_config or {},
        }
        return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()

    def _scalar_key(
        self,
        *,
        kind: str,
        fen: str,
        chess960: bool,
        depth: int,
        move_uci: str | None = None,
        pov: str = "side_to_move",
        engine_config: Mapping[str, Any] | None = None,
    ) -> str:
        payload = {
            "version": _CACHE_VERSION,
            "kind": kind,
            "fen": fen,
            "chess960": bool(chess960),
            "depth": int(depth),
            "move_uci": move_uci,
            "pov": pov,
            "engine_config": engine_config or {},
        }
        return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def analyze_multipv(
    engine: Any,
    fen: str,
    *,
    depth: int,
    k: int,
    pv_len: int,
    cache: SqliteMultipvCache | None = None,
    chess960: bool = False,
    engine_config: Mapping[str, Any] | None = None,
) -> MultipvAnalysis:
    """Analyze ``fen`` with Stockfish MultiPV and optional SQLite caching."""

    if cache is not None:
        cached = cache.get(
            fen=fen,
            chess960=chess960,
            depth=depth,
            k=k,
            pv_len=pv_len,
            engine_config=engine_config,
        )
        if cached is not None:
            return cached

    board = chess.Board(fen, chess960=chess960)
    if not board.is_valid():
        analysis = MultipvAnalysis(
            fen=fen,
            chess960=chess960,
            depth=depth,
            k=k,
            pv_len=pv_len,
            moves=[],
            engine_metadata=_engine_metadata(engine, engine_config),
        )
        if cache is not None:
            cache.put(analysis, engine_config=engine_config)
        return analysis

    info = engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
        multipv=max(1, int(k)),
    )
    rows = info if isinstance(info, list) else [info]
    moves: list[MultipvMove] = []
    for rank, row in enumerate(rows[:k], start=1):
        parsed = _parse_multipv_row(row, board, rank=rank, pv_len=pv_len)
        if parsed is not None:
            moves.append(parsed)
    analysis = MultipvAnalysis(
        fen=fen,
        chess960=chess960,
        depth=depth,
        k=k,
        pv_len=pv_len,
        moves=moves,
        engine_metadata=_engine_metadata(engine, engine_config),
    )
    if cache is not None:
        cache.put(analysis, engine_config=engine_config)
    return analysis


def score_move_wpd(
    engine: Any,
    fen: str,
    move_uci: str,
    *,
    depth: int,
    k: int,
    pv_len: int,
    cache: SqliteMultipvCache | None = None,
    chess960: bool = False,
    engine_config: Mapping[str, Any] | None = None,
) -> WpdDiagnostics:
    """Score ``move_uci`` by WPD, reusing MultiPV rows before post-move search."""

    analysis = analyze_multipv(
        engine,
        fen,
        depth=depth,
        k=k,
        pv_len=pv_len,
        cache=cache,
        chess960=chess960,
        engine_config=engine_config,
    )
    analysis_cache_hit = bool(cache.last_hit) if cache is not None else False
    best = analysis.moves[0] if analysis.moves else None
    best_expectation = best.expectation if best is not None else None
    predicted = next((move for move in analysis.moves if move.uci == move_uci), None)
    if predicted is not None:
        return _wpd_result(
            predicted_move=move_uci,
            best_move=best.uci if best is not None else None,
            best_expectation=best_expectation,
            predicted_expectation=predicted.expectation,
            multipv_hit=True,
            postmove_hit=False,
            cache_hit=analysis_cache_hit,
        )

    try:
        board = chess.Board(fen, chess960=chess960)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return _wpd_result(
                predicted_move=move_uci,
                best_move=best.uci if best is not None else None,
                best_expectation=best_expectation,
                predicted_expectation=0.0,
                multipv_hit=False,
                postmove_hit=False,
                cache_hit=analysis_cache_hit,
            )
        board.push(move)
    except (ValueError, TypeError, AssertionError):
        return _wpd_result(
            predicted_move=move_uci,
            best_move=best.uci if best is not None else None,
            best_expectation=best_expectation,
            predicted_expectation=0.0,
            multipv_hit=False,
            postmove_hit=False,
            cache_hit=analysis_cache_hit,
        )

    terminal = _terminal_expectation_for_previous_side(board)
    if terminal is not None:
        predicted_expectation = terminal
    else:
        reply_analysis = analyze_multipv(
            engine,
            board.fen(),
            depth=depth,
            k=1,
            pv_len=pv_len,
            cache=cache,
            chess960=chess960,
            engine_config=engine_config,
        )
        if reply_analysis.moves:
            predicted_expectation = 1.0 - reply_analysis.moves[0].expectation
        else:
            predicted_expectation = 0.5

    return _wpd_result(
        predicted_move=move_uci,
        best_move=best.uci if best is not None else None,
        best_expectation=best_expectation,
        predicted_expectation=predicted_expectation,
        multipv_hit=False,
        postmove_hit=True,
        cache_hit=analysis_cache_hit,
    )


def _parse_multipv_row(
    row: Mapping[str, Any],
    board: chess.Board,
    *,
    rank: int,
    pv_len: int,
) -> MultipvMove | None:
    pv_value = row.get("pv", [])
    pv = [move.uci() for move in pv_value if isinstance(move, chess.Move)]
    if not pv:
        return None
    score = row.get("score")
    cp, mate = _side_to_move_score(score, board.turn)
    expectation = _expectation(cp=cp, mate=mate)
    return MultipvMove(
        uci=pv[0],
        rank=rank,
        cp=cp,
        mate=mate,
        expectation=expectation,
        pv=pv[:pv_len] if pv_len > 0 else [],
    )


def _side_to_move_score(
    score: Any,
    turn: chess.Color,
) -> tuple[int | None, int | None]:
    if hasattr(score, "pov"):
        score = score.pov(turn)
    elif hasattr(score, "white"):
        score = score.white() if turn == chess.WHITE else score.black()
    if score is None:
        return None, None
    is_mate = getattr(score, "is_mate", lambda: False)()
    if is_mate:
        return None, int(score.mate())
    value = score.score() if hasattr(score, "score") else None
    return (int(value) if value is not None else None), None


def _expectation(*, cp: int | None, mate: int | None) -> float:
    if mate is not None:
        if mate > 0:
            return 1.0
        if mate < 0:
            return 0.0
        return 0.5
    if cp is None:
        return 0.5
    # Smooth, cheap Stockfish-like centipawn to expectation transform.
    return 1.0 / (1.0 + math.exp(-float(cp) / 300.0))


def _wpd_result(
    *,
    predicted_move: str,
    best_move: str | None,
    best_expectation: float | None,
    predicted_expectation: float | None,
    multipv_hit: bool,
    postmove_hit: bool,
    cache_hit: bool,
) -> WpdDiagnostics:
    if best_expectation is None or predicted_expectation is None:
        wpd = None
        reward = 0.0
    else:
        wpd = max(0.0, best_expectation - predicted_expectation)
        reward = max(0.0, min(1.0, 1.0 - wpd))
    return WpdDiagnostics(
        predicted_move=predicted_move,
        best_move=best_move,
        best_expectation=best_expectation,
        predicted_expectation=predicted_expectation,
        wpd=wpd,
        reward=reward,
        reward_bucket=_reward_bucket(wpd),
        multipv_hit=multipv_hit,
        postmove_hit=postmove_hit,
        cache_hit=cache_hit,
    )


def _reward_bucket(wpd: float | None) -> str:
    if wpd is None:
        return "unscored"
    if wpd <= 0.005:
        return "best"
    if wpd <= 0.03:
        return "near_best"
    if wpd <= 0.10:
        return "playable"
    if wpd <= 0.25:
        return "weak"
    return "bad"


def _terminal_expectation_for_previous_side(board: chess.Board) -> float | None:
    outcome = board.outcome()
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.5
    previous_side = not board.turn
    return 1.0 if outcome.winner == previous_side else 0.0


def _engine_metadata(
    engine: Any,
    engine_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"engine_config": dict(engine_config or {})}
    name = stockfish_engine_name(engine)
    if name:
        metadata["engine_name"] = name
    return metadata


def _analysis_payload(analysis: MultipvAnalysis) -> dict[str, Any]:
    return {
        "fen": analysis.fen,
        "chess960": analysis.chess960,
        "depth": analysis.depth,
        "k": analysis.k,
        "pv_len": analysis.pv_len,
        "moves": [asdict(move) for move in analysis.moves],
        "engine_metadata": analysis.engine_metadata,
    }


def _analysis_from_payload(payload: Mapping[str, Any]) -> MultipvAnalysis:
    return MultipvAnalysis(
        fen=str(payload["fen"]),
        chess960=bool(payload["chess960"]),
        depth=int(payload["depth"]),
        k=int(payload["k"]),
        pv_len=int(payload["pv_len"]),
        moves=[MultipvMove(**move) for move in payload.get("moves", [])],
        engine_metadata=dict(payload.get("engine_metadata", {})),
    )


def _stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


__all__ = [
    "MultipvAnalysis",
    "MultipvMove",
    "SqliteMultipvCache",
    "WpdDiagnostics",
    "analyze_multipv",
    "score_move_wpd",
]
