"""
Eval-set decontamination: ensure no training FEN appears in the eval blocklist.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def check_no_contamination(fen: str, blocklist: frozenset[str]) -> bool:
    """Return True if *fen* is NOT in the eval blocklist (i.e., safe)."""
    return fen not in blocklist


def audit_output_files(
    output_dir: str, blocklist: frozenset[str]
) -> dict[str, list[str]]:
    """Scan all JSONL files under *output_dir* for contaminated FENs.

    Returns ``{filename: [contaminated_fens]}``.  Empty dict means clean.
    Also logs a warning for any files containing malformed JSON lines.
    """
    report: dict[str, list[str]] = {}
    for jsonl_path in Path(output_dir).rglob("*.jsonl"):
        contaminated: list[str] = []
        malformed = 0
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                fen = obj.get("fen", "")
                if fen and fen in blocklist:
                    contaminated.append(fen)
        if malformed:
            logger.warning(
                "Decontamination audit: %s has %d malformed JSON line(s)",
                jsonl_path, malformed,
            )
        if contaminated:
            report[str(jsonl_path)] = contaminated
    return report
