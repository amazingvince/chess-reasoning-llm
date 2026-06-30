"""Audit generated SFT rows against held-out eval FEN blocklists."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from chess_llm.core.board import canonical_fen_key, variant_fen_key
from chess_llm.sft.context import raw_is_chess960
from chess_llm.sft.files import iter_jsonl_artifacts

logger = logging.getLogger(__name__)


def check_no_contamination(
    fen: str,
    blocklist: frozenset[str],
    *,
    chess960: bool = False,
) -> bool:
    """Return True when *fen* is absent from the eval blocklist."""
    return (
        fen not in blocklist
        and variant_fen_key(fen, chess960=chess960) not in blocklist
        and canonical_fen_key(fen, chess960=chess960) not in blocklist
    )


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
                fen = row.get("fen", "")
                if fen and not check_no_contamination(
                    fen,
                    blocklist,
                    chess960=raw_is_chess960(row),
                ):
                    contaminated.append(fen)
        if malformed:
            logger.warning(
                "Decontamination audit: %s has %d malformed JSON line(s)",
                jsonl_path,
                malformed,
            )
        if contaminated:
            report[str(jsonl_path)] = contaminated
    return report


__all__ = ["audit_output_files", "check_no_contamination"]
