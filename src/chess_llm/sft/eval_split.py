"""Eval split generation and FEN blocklist management."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from random import Random

import chess

from chess_llm.core.board import canonical_fen_key, variant_fen_key
from chess_llm.sft.context import raw_fen_identity_key, raw_is_chess960
from chess_llm.sft.settings import DEFAULT_EVAL_SPLIT_SIZES, DEFAULT_MASTER_SEED

logger = logging.getLogger(__name__)

EVAL_SPLIT_SIZES: dict[str, int] = dict(DEFAULT_EVAL_SPLIT_SIZES)

RULES_SPLIT_STATE_FRACTIONS: dict[str, float] = {
    "check": 0.10,
    "checkmate": 0.05,
    "stalemate": 0.05,
}


def effective_eval_split_sizes(
    split_sizes: dict[str, int] | None = None,
    *,
    volume_override: int | None = None,
) -> dict[str, int]:
    """Return eval split targets, capped for smoke runs when requested."""
    targets = dict(EVAL_SPLIT_SIZES if split_sizes is None else split_sizes)
    if volume_override is None:
        return targets
    return {
        split_name: min(target_size, volume_override)
        for split_name, target_size in targets.items()
    }


def generate_all_eval_splits(
    sources: dict[str, list[dict]],
    seed: int = DEFAULT_MASTER_SEED,
    *,
    split_sizes: dict[str, int] | None = None,
) -> dict[str, list[dict]]:
    """Create held-out eval splits without cross-split FEN overlap."""
    rng = Random(seed)
    targets = split_sizes if split_sizes is not None else EVAL_SPLIT_SIZES
    splits: dict[str, list[dict]] = {}
    used_fens: set[str] = set()

    for split_name, target_size in targets.items():
        candidates = sources.get(split_name, [])
        if not candidates:
            logger.warning("No source data for eval split %r - skipping", split_name)
            splits[split_name] = []
            continue

        seen_in_candidates: set[str] = set()
        available: list[dict] = []
        for candidate in candidates:
            fen = candidate.get("fen", "")
            fen_key = _example_fen_key(candidate)
            if not fen or fen_key in used_fens or fen_key in seen_in_candidates:
                continue
            seen_in_candidates.add(fen_key)
            available.append(candidate)

        sampled = rng.sample(available, min(target_size, len(available)))
        if split_name == "rules":
            sampled = _stratify_rules_split(sampled, available, rng)
        splits[split_name] = sampled
        used_fens.update(
            _example_fen_key(example)
            for example in sampled
            if example.get("fen")
        )

        logger.info(
            "Eval split %r: %d / %d target (%d candidates after dedup)",
            split_name,
            len(sampled),
            target_size,
            len(available),
        )

    return splits


def build_blocklist(
    eval_splits: dict[str, list[dict]],
    fen_pool: list[dict] | None = None,
) -> frozenset[str]:
    """Extract canonical FEN keys from eval splits.

    When *fen_pool* is given, every pool position sharing a sampled
    position's ``game_id`` is blocked too, so same-game neighbor plies never
    leak into training. Rows without ``game_id`` fall back to exact-position
    blocking.
    """
    fens: set[str] = set()
    sampled_game_ids: set[str] = set()
    for split_examples in eval_splits.values():
        for example in split_examples:
            if example.get("fen"):
                fens.add(_example_fen_key(example))
                game_id = example.get("game_id")
                if game_id:
                    sampled_game_ids.add(str(game_id))
    if fen_pool and sampled_game_ids:
        neighbors = 0
        for entry in fen_pool:
            if not isinstance(entry, dict) or not entry.get("fen"):
                continue
            game_id = entry.get("game_id")
            if game_id and str(game_id) in sampled_game_ids:
                key = _example_fen_key(entry)
                if key not in fens:
                    fens.add(key)
                    neighbors += 1
        if neighbors:
            logger.info(
                "Blocked %d same-game neighbor positions for %d eval games",
                neighbors,
                len(sampled_game_ids),
            )
    logger.info("Built eval blocklist with %d FENs", len(fens))
    return frozenset(fens)


def save_eval_splits(
    splits: dict[str, list[dict]],
    path: str | Path,
    *,
    fen_pool: list[dict] | None = None,
    extra_blocklist_keys: frozenset[str] | set[str] = frozenset(),
) -> None:
    """Save eval splits as one JSONL file per split plus a blocklist."""
    base = Path(path)
    base.mkdir(parents=True, exist_ok=True)
    for name, examples in splits.items():
        out = base / f"{name}.jsonl"
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            for example in examples:
                fh.write(json.dumps(example, ensure_ascii=False) + "\n")
        logger.info("Saved eval split %r (%d examples) to %s", name, len(examples), out)

    blocklist_path = base / "blocklist.txt"
    keys = set(build_blocklist(splits, fen_pool)) | set(extra_blocklist_keys)
    with blocklist_path.open("w", encoding="utf-8", newline="\n") as fh:
        for fen in sorted(keys):
            fh.write(fen + "\n")
    logger.info("Saved blocklist to %s", blocklist_path)


def load_blocklist(path: str | Path) -> frozenset[str]:
    """Load a text blocklist and normalize each FEN key."""
    fens: set[str] = set()
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            fen = line.strip()
            if fen:
                fens.add(_normalize_blocklist_key(fen))
    return frozenset(fens)


def partition_eco_codes(
    openings: list[dict],
    holdout_fraction: float = 0.15,
    seed: int = DEFAULT_MASTER_SEED,
) -> tuple[list[dict], list[dict]]:
    """Split openings by ECO code so same-family positions stay together."""
    rng = Random(seed)
    eco_to_rows: dict[str, list[dict]] = {}
    missing_eco_rows: list[dict] = []
    for row in openings:
        eco = row.get("eco", "")
        if eco:
            eco_to_rows.setdefault(eco, []).append(row)
        else:
            missing_eco_rows.append(row)

    ecos = sorted(eco_to_rows)
    rng.shuffle(ecos)
    if holdout_fraction <= 0 or not ecos:
        n_holdout = 0
    else:
        n_holdout = min(len(ecos), max(1, int(len(ecos) * holdout_fraction)))
    eval_ecos = set(ecos[:n_holdout])

    eval_openings: list[dict] = []
    train_openings: list[dict] = []
    for eco in ecos:
        rows = eco_to_rows[eco]
        if eco in eval_ecos:
            eval_openings.extend(rows)
        else:
            train_openings.extend(rows)
    train_openings.extend(missing_eco_rows)
    return eval_openings, train_openings


def _classify_board_state(example: dict) -> str | None:
    """Classify a row as checkmate/stalemate/check/quiet, or None if unparseable."""
    fen = example.get("fen", "")
    if not fen:
        return None
    try:
        board = chess.Board(str(fen), chess960=raw_is_chess960(example))
    except (TypeError, ValueError):
        return None
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_check():
        return "check"
    return "quiet"


def _stratify_rules_split(
    sampled: list[dict],
    available: list[dict],
    rng: Random,
) -> list[dict]:
    """Boost check/checkmate/stalemate coverage in the rules eval split.

    Targets RULES_SPLIT_STATE_FRACTIONS best-effort from the available pool
    so check_detection eval is not dominated by quiet positions.
    """
    if not sampled:
        return sampled

    result = list(sampled)
    states = [_classify_board_state(example) for example in result]
    counts: dict[str, int] = {}
    for state in states:
        if state is not None:
            counts[state] = counts.get(state, 0) + 1

    deficits = {
        state: max(
            0,
            math.ceil(fraction * len(result)) - counts.get(state, 0),
        )
        for state, fraction in RULES_SPLIT_STATE_FRACTIONS.items()
    }
    if not any(deficits.values()):
        return result

    sampled_keys = {_example_fen_key(example) for example in result}
    candidates = list(available)
    rng.shuffle(candidates)

    quiet_indices = [
        index for index, state in enumerate(states) if state == "quiet"
    ]
    rng.shuffle(quiet_indices)

    for candidate in candidates:
        if not any(deficits.values()) or not quiet_indices:
            break
        state = _classify_board_state(candidate)
        if state is None or deficits.get(state, 0) <= 0:
            continue
        key = _example_fen_key(candidate)
        if key in sampled_keys:
            continue
        result[quiet_indices.pop()] = candidate
        sampled_keys.add(key)
        deficits[state] -= 1

    unmet = {state: deficit for state, deficit in deficits.items() if deficit > 0}
    if unmet:
        logger.warning(
            "RULES EVAL SPLIT IS UNDER-STRATIFIED: pool lacks positions for %s "
            "(check_detection eval will skew toward quiet positions)",
            ", ".join(f"{state} (short {deficit})" for state, deficit in sorted(unmet.items())),
        )
    return result


def _example_fen_key(example: dict) -> str:
    return raw_fen_identity_key(example)


def _example_is_chess960(example: dict) -> bool:
    return raw_is_chess960(example)


def _normalize_blocklist_key(value: str) -> str:
    if value.startswith("std:"):
        return variant_fen_key(value.removeprefix("std:"), chess960=False)
    if value.startswith("960:"):
        return variant_fen_key(value.removeprefix("960:"), chess960=True)
    return variant_fen_key(value)


__all__ = [
    "EVAL_SPLIT_SIZES",
    "build_blocklist",
    "canonical_fen_key",
    "effective_eval_split_sizes",
    "generate_all_eval_splits",
    "load_blocklist",
    "partition_eco_codes",
    "save_eval_splits",
]
