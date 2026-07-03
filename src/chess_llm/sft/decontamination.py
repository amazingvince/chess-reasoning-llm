"""Audit generated SFT rows against held-out eval FEN blocklists."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterator

from chess_llm.core.board import variant_fen_key
from chess_llm.sft.context import raw_is_chess960
from chess_llm.sft.files import iter_jsonl_artifacts

logger = logging.getLogger(__name__)

_FEN_RE = re.compile(
    r"[pnbrqkPNBRQK1-8]+(?:/[pnbrqkPNBRQK1-8]+){7}"
    r" [wb] (?:-|[KQkqA-Ha-h]+) (?:-|[a-h][1-8])(?: \d+ \d+)?"
)


def check_no_contamination(
    fen: str,
    blocklist: frozenset[str],
    *,
    chess960: bool = False,
) -> bool:
    """Return True when *fen* is absent from the eval blocklist."""
    primary = variant_fen_key(fen, chess960=chess960)
    if primary in blocklist:
        return False
    # A castle-less position is the same position under standard and
    # Chess960 rules, so it must be blocked across both variant pools.
    prefix, _, body = primary.partition(":")
    parts = body.split()
    if len(parts) >= 3 and parts[2] == "-":
        other_prefix = "std" if prefix == "960" else "960"
        if f"{other_prefix}:{body}" in blocklist:
            return False
    return True


def _iter_row_fens(row: dict) -> Iterator[str]:
    """Yield the known FEN-bearing fields of a generated SFT row."""
    fen = row.get("fen", "")
    if fen:
        yield str(fen)
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if "fen" not in key.lower() or not isinstance(value, str):
                continue
            if _FEN_RE.fullmatch(value.strip()):
                yield value.strip()
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str):
                yield from _FEN_RE.findall(content)


def row_contaminated_fens(row: dict, blocklist: frozenset[str]) -> list[str]:
    """Return every FEN in *row* (start, metadata, answers) that is blocked."""
    chess960 = raw_is_chess960(row)
    return [
        fen
        for fen in _iter_row_fens(row)
        if not check_no_contamination(fen, blocklist, chess960=chess960)
    ]


def check_row_no_contamination(row: dict, blocklist: frozenset[str]) -> bool:
    """Return True when no FEN-bearing field of *row* is blocklisted."""
    return not row_contaminated_fens(row, blocklist)


def audit_output_files(
    output_dir: str | Path,
    blocklist: frozenset[str],
) -> dict[str, list[str]]:
    """Scan JSONL files under *output_dir* for rows whose FENs are blocklisted."""
    report: dict[str, list[str]] = {}
    for jsonl_path in iter_jsonl_artifacts(output_dir):
        contaminated: list[str] = []
        malformed = 0
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                hits = row_contaminated_fens(row, blocklist)
                if hits:
                    contaminated.append(hits[0])
        if malformed:
            logger.warning(
                "Decontamination audit: %s has %d malformed JSON line(s)",
                jsonl_path,
                malformed,
            )
        if contaminated:
            report[str(jsonl_path)] = contaminated
    return report


__all__ = [
    "audit_output_files",
    "check_no_contamination",
    "check_row_no_contamination",
    "row_contaminated_fens",
]
