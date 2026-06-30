"""Backend configuration for the Chess LLM workbench."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class BackendSettings:
    """Runtime settings for the UI backend."""

    artifact_root: Path
    book_root: Path
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_mode: str = "auto"
    llm_model: str = "local-chess-llm"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 512
    stockfish_path: Path | None = None
    stockfish_depth: int = 16
    stockfish_threads: int = 1
    stockfish_hash_mb: int = 256
    tool_batch_limit: int = 100

    @classmethod
    def from_env(cls) -> "BackendSettings":
        root = _repo_root()
        return cls(
            artifact_root=Path(os.getenv("CHESS_UI_ARTIFACT_ROOT", root / "ui" / "runs")),
            book_root=Path(
                os.getenv(
                    "CHESS_UI_BOOK_ROOT",
                    root / "sft" / "make_data" / "polyglot_opening_books",
                )
            ),
            llm_base_url=os.getenv("CHESS_UI_LLM_BASE_URL"),
            llm_api_key=os.getenv("CHESS_UI_LLM_API_KEY"),
            llm_mode=os.getenv("CHESS_UI_LLM_MODE", "auto"),
            llm_model=os.getenv("CHESS_UI_LLM_MODEL", "local-chess-llm"),
            llm_temperature=float(os.getenv("CHESS_UI_LLM_TEMPERATURE", "0.2")),
            llm_max_tokens=int(os.getenv("CHESS_UI_LLM_MAX_TOKENS", "512")),
            stockfish_path=(
                Path(path)
                if (path := os.getenv("CHESS_UI_STOCKFISH_PATH"))
                else None
            ),
            stockfish_depth=int(os.getenv("CHESS_UI_STOCKFISH_DEPTH", "16")),
            stockfish_threads=int(os.getenv("CHESS_UI_STOCKFISH_THREADS", "1")),
            stockfish_hash_mb=int(os.getenv("CHESS_UI_STOCKFISH_HASH_MB", "256")),
            tool_batch_limit=int(os.getenv("CHESS_UI_TOOL_BATCH_LIMIT", "100")),
        )
