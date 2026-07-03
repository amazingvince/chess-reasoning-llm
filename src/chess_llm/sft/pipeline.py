"""Package-owned orchestration for SFT data generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from random import Random
from typing import Mapping, Sequence

from chess_llm.evals.benchmark import freeze_and_save
from chess_llm.sft.completeness import audit_output_completeness
from chess_llm.sft.context import raw_fen_identity_key
from chess_llm.sft.decontamination import (
    audit_output_files,
    check_row_no_contamination,
)
from chess_llm.sft.eval_split import (
    build_blocklist,
    effective_eval_split_sizes,
    generate_all_eval_splits,
    load_blocklist,
    save_eval_splits,
)
from chess_llm.sft.files import iter_jsonl_artifacts
from chess_llm.sft.fen_pool import FENPool
from chess_llm.sft.generators import TIER_GENERATORS as PACKAGE_TIER_GENERATORS
from chess_llm.sft.output import JSONLWriter, PipelineStats
from chess_llm.sft.settings import (
    DEFAULT_CHESS960_RATIOS,
    DEFAULT_VOLUMES,
    SftDataSettings,
)
from chess_llm.sft.source_preparation import (
    build_eval_split_sources,
    reserve_training_rows_for_volume,
)
from chess_llm.sft.source_readiness import (
    build_source_readiness_report,
    eval_splits_for_tiers,
    write_source_readiness_report,
)
from chess_llm.sft.sources import (
    extract_game_positions,
    load_mate,
    load_openings,
    load_puzzles,
    load_self_play_positions,
    sample_chess960_positions,
    stream_evals,
    stream_games,
)
from chess_llm.sft.validation import validate_example

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_MAKE_DATA_ROOT = _REPO_ROOT / "sft" / "make_data"
_SETTINGS_ROOT = _LEGACY_MAKE_DATA_ROOT if _LEGACY_MAKE_DATA_ROOT.exists() else Path.cwd()

SETTINGS = SftDataSettings.from_env(_SETTINGS_ROOT)

MASTER_SEED = SETTINGS.master_seed
OUTPUT_DIR = SETTINGS.output_dir
POOL_DIR = SETTINGS.pool_dir
EVAL_SPLITS_DIR = SETTINGS.eval_splits_dir
ANNOTATIONS_DIR = SETTINGS.annotations_dir
TIER_OUTPUT_DIR = SETTINGS.tier_output_dir
BENCHMARK_DIR = SETTINGS.benchmark_dir
POLYGLOT_DIR = SETTINGS.polyglot_dir
SYZYGY_PATH = SETTINGS.syzygy_path
MIN_DEPTH_TRAINING = SETTINGS.min_depth_training
MIN_DEPTH_BESTMOVE = SETTINGS.min_depth_bestmove
MIN_DEPTH_EVAL_BENCHMARK = SETTINGS.min_depth_eval_benchmark
EVAL_SPLIT_SIZES = dict(SETTINGS.eval_split_sizes)
SELF_PLAY_DIR = SETTINGS.self_play_dir
SELF_PLAY_MAX_POSITIONS = SETTINGS.self_play_max_positions
SELF_PLAY_RATIO = SETTINGS.self_play_ratio
VOLUME_LICHESS_GAME_DATA_FILES = tuple(SETTINGS.volume_lichess_game_data_files)
EVAL_SPLIT_MANIFEST_NAME = "manifest.json"
EVAL_SPLIT_FINGERPRINT_CACHE_NAME = "source_fingerprints.cache.json"


def _load_generator_registry() -> dict[int, list[type]]:
    return {tier: list(generators) for tier, generators in PACKAGE_TIER_GENERATORS.items()}


TIER_GENERATORS: dict[int, list[type]] = _load_generator_registry()


def _pipeline_stats_for_volume_override(volume_override: int | None) -> PipelineStats:
    """Build generation stats whose target column reflects the active volume."""
    if volume_override is None:
        return PipelineStats()
    return PipelineStats(
        volumes={task_id: volume_override for task_id in DEFAULT_VOLUMES}
    )


def chess960_source_target_count(
    *,
    n_standard: int,
    volume_override: int | None,
) -> int:
    """Return Chess960 source rows needed for train mix plus eval reserve."""
    if volume_override is None:
        target_960_ratio = 0.15
        target = int(n_standard * target_960_ratio / (1 - target_960_ratio))
        return max(5_000, min(target, 100_000))

    target_960_ratio = max(DEFAULT_CHESS960_RATIOS.values())
    train_mix_target = int(n_standard * target_960_ratio / (1 - target_960_ratio))
    eval_reserve = effective_eval_split_sizes(
        EVAL_SPLIT_SIZES,
        volume_override=volume_override,
    ).get("chess960", 0)
    return max(volume_override + eval_reserve, train_mix_target + eval_reserve)


def lichess_game_data_files_for_volume(
    volume_override: int | None,
) -> list[str] | None:
    """Restrict expensive game-shard discovery for bounded smoke generation."""
    if volume_override is None:
        return None
    return list(VOLUME_LICHESS_GAME_DATA_FILES)


def load_sources(volume_override: int | None = None) -> dict:
    """Load data sources and build the shared generator config."""
    logger.info("Loading data sources...")
    rng = Random(MASTER_SEED)
    max_items = volume_override * 10 if volume_override else 500_000
    config: dict = {}

    logger.info("Loading Lichess games...")
    fen_pool_entries: list[dict] = []
    game_positions: list[dict] = []
    lichess_game_data_files = lichess_game_data_files_for_volume(volume_override)
    try:
        for game in stream_games(
            max_games=max_items,
            data_files=lichess_game_data_files,
        ):
            for pos in extract_game_positions(game):
                entry = {
                    "fen": pos["fen"],
                    "source": "lichess_games",
                    "game_phase": pos["game_phase"],
                    "is_chess960": False,
                }
                if pos.get("game_id"):
                    entry["game_id"] = pos["game_id"]
                fen_pool_entries.append(entry)
                game_positions.append(pos)
                if len(fen_pool_entries) >= max_items:
                    break
            if len(fen_pool_entries) >= max_items:
                break
    except Exception as exc:  # pragma: no cover - depends on external datasets
        logger.warning("Lichess games loading failed: %s", exc)

    logger.info(
        "Lichess games: %d FENs, %d game_positions",
        len(fen_pool_entries),
        len(game_positions),
    )

    logger.info("Loading Lichess puzzles...")
    puzzles: list[dict] = []
    try:
        puzzles.extend(load_puzzles(max_puzzles=max_items))
    except Exception as exc:  # pragma: no cover - depends on external datasets
        logger.warning("Lichess puzzles loading failed: %s", exc)

    logger.info("Loading Lichess openings...")
    openings: list[dict] = []
    try:
        for opening in load_openings(max_openings=max_items):
            openings.append(opening)
            fen_pool_entries.append(
                {
                    "fen": opening["fen"],
                    "source": "lichess_openings",
                    "is_chess960": False,
                }
            )
    except Exception as exc:  # pragma: no cover - depends on external datasets
        logger.warning("Lichess openings loading failed: %s", exc)

    logger.info("Loading Lichess position evaluations...")
    position_evals: list[dict] = []
    best_move_evals: list[dict] = []
    try:
        for ev in stream_evals(min_depth=MIN_DEPTH_TRAINING, max_rows=max_items):
            position_evals.append(ev)
            if ev.get("depth", 0) >= MIN_DEPTH_BESTMOVE:
                best_move_evals.append(ev)
    except Exception as exc:  # pragma: no cover - depends on external datasets
        logger.warning("Lichess evals loading failed: %s", exc)

    # The eval cache streams rows in depth order; shuffle so consumers do not
    # see a depth-sorted slice (fen_pool_entries gets the same treatment below).
    rng.shuffle(position_evals)
    rng.shuffle(best_move_evals)

    for ev in position_evals:
        fen_pool_entries.append(
            {
                "fen": ev["fen"],
                "source": "lichess_evals",
                "game_phase": "unknown",
                "is_chess960": False,
            }
        )
    logger.info(
        "Added %d lichess eval FENs to pool (total: %d)",
        len(position_evals),
        len(fen_pool_entries),
    )

    if len(game_positions) < 50_000:
        supplement_count = 0
        for ev in position_evals:
            if ev.get("fen") and ev.get("best_move"):
                game_positions.append(
                    {
                        "fen": ev["fen"],
                        "move_played_uci": ev["best_move"],
                        "game_phase": "unknown",
                        "material_balance": 0,
                        "ply": 0,
                    }
                )
                supplement_count += 1
        logger.info(
            "Supplemented game_positions with %d eval entries (total: %d)",
            supplement_count,
            len(game_positions),
        )

    logger.info("Loading self-play positions...")
    self_play_positions: list[dict] = []
    try:
        self_play_positions = load_self_play_positions(SELF_PLAY_DIR)
    except Exception as exc:  # pragma: no cover - depends on local harvest runs
        logger.warning("Self-play position loading failed: %s", exc)

    if self_play_positions:
        from chess_llm.core.board import is_legal_move

        self_play_cap = min(
            SELF_PLAY_MAX_POSITIONS,
            int(SELF_PLAY_RATIO * len(fen_pool_entries)),
        )
        if len(self_play_positions) > self_play_cap:
            self_play_positions = (
                rng.sample(self_play_positions, self_play_cap)
                if self_play_cap > 0
                else []
            )
        added_self_play = 0
        for pos in self_play_positions:
            if not is_legal_move(pos["fen"], pos["move_played_uci"]):
                logger.warning(
                    "Skipping self-play position with illegal move %s in %s",
                    pos["move_played_uci"],
                    pos["fen"],
                )
                continue
            fen_pool_entries.append(
                {
                    "fen": pos["fen"],
                    "source": "self_play",
                    "game_phase": pos["game_phase"],
                    "is_chess960": False,
                    "game_id": pos["game_id"],
                }
            )
            game_positions.append(pos)
            added_self_play += 1
        logger.info(
            "Added %d self-play FENs to pool (cap: %d, total: %d)",
            added_self_play,
            self_play_cap,
            len(fen_pool_entries),
        )

    logger.info("Loading Polyglot opening books...")
    import chess
    from chess_llm.sft.sources.polyglot_books import get_weighted_moves, load_book

    book_moves: dict[str, dict[str, float]] = {}
    polyglot_dir = Path(POLYGLOT_DIR)
    if polyglot_dir.exists():
        for bin_file in polyglot_dir.glob("*.bin"):
            try:
                with load_book(str(bin_file)) as reader:
                    for opening in openings:
                        board = chess.Board(opening["fen"])
                        moves = get_weighted_moves(reader, board)
                        if not moves:
                            continue
                        # Each book uses its own weight scale; normalize per
                        # position so merged weights reflect relative popularity.
                        book_total = sum(weight for _, weight in moves)
                        if book_total <= 0:
                            continue
                        merged = book_moves.setdefault(opening["fen"], {})
                        for uci, weight in moves:
                            merged[uci] = merged.get(uci, 0.0) + weight / book_total
            except Exception as exc:  # pragma: no cover - depends on local book files
                logger.warning("Polyglot book %s failed: %s", bin_file, exc)

    sorted_book_moves: dict[str, list[tuple[str, float]]] = {}
    for fen, move_weights in book_moves.items():
        sorted_book_moves[fen] = sorted(
            move_weights.items(),
            key=lambda item: item[1],
            reverse=True,
        )

    endgame_positions: list[dict] = []
    syzygy_path = Path(SYZYGY_PATH)
    if syzygy_path.exists():
        logger.info("Loading Syzygy endgame positions...")
        from chess_llm.sft.sources.syzygy_probing import (
            MATERIAL_CONFIGS,
            best_dtz_move,
            open_tablebase,
            sample_endgame_positions,
        )

        try:
            with open_tablebase(str(syzygy_path)) as tb:
                for mat in MATERIAL_CONFIGS:
                    n = volume_override or 3000
                    for pos in sample_endgame_positions(tb, mat, n, rng):
                        board = chess.Board(pos["fen"])
                        pos["best_move"] = best_dtz_move(tb, board) or ""
                        endgame_positions.append(pos)
        except Exception as exc:  # pragma: no cover - depends on local tablebases
            logger.warning("Syzygy loading failed: %s", exc)

    logger.info("Generating Chess960 positions...")
    n_chess960 = chess960_source_target_count(
        n_standard=len(fen_pool_entries),
        volume_override=volume_override,
    )
    logger.info(
        "Chess960 target: %d (standard pool: %d)",
        n_chess960,
        len(fen_pool_entries),
    )

    chess960_positions = sample_chess960_positions(
        n=n_chess960,
        n_random_moves=20,
        rng=rng,
    )
    fen_pool_entries.extend(chess960_positions)

    logger.info("Loading MATE dataset...")
    mate_rows: list[dict] = []
    try:
        mate_rows.extend(load_mate(max_rows=max_items))
    except Exception as exc:  # pragma: no cover - depends on external datasets
        logger.warning("MATE dataset loading failed: %s", exc)

    logger.info(
        "Deduplicating FEN pool via FENPool (%d raw entries)...",
        len(fen_pool_entries),
    )
    pool = FENPool()
    for entry in fen_pool_entries:
        fen = entry["fen"] if isinstance(entry, dict) else entry
        tags = entry if isinstance(entry, dict) else {"fen": fen}
        pool.add(fen, **{k: v for k, v in tags.items() if k != "fen"})
    logger.info("FEN pool after dedup: %d unique FENs", len(pool))

    fen_pool_entries = pool.all_rows()
    rng.shuffle(fen_pool_entries)

    config["fen_pool"] = fen_pool_entries
    config["game_positions"] = game_positions
    config["puzzles"] = puzzles
    config["openings"] = openings
    config["position_evals"] = position_evals
    config["best_move_evals"] = best_move_evals
    config["consequence_evals"] = position_evals
    config["candidate_rating_evals"] = [
        ev
        for ev in position_evals
        if isinstance(ev.get("candidate_ratings"), list)
        or isinstance(ev.get("move_evaluations"), list)
    ]
    config["book_moves"] = sorted_book_moves
    config["endgame_positions"] = endgame_positions
    config["mate_rows"] = mate_rows

    logger.info(
        "Sources loaded: %d FENs, %d puzzles, %d openings, %d evals, "
        "%d endgames, %d MATE rows",
        len(fen_pool_entries),
        len(puzzles),
        len(openings),
        len(position_evals),
        len(endgame_positions),
        len(mate_rows),
    )

    if not fen_pool_entries:
        logger.error(
            "CRITICAL: fen_pool is empty after loading all sources. "
            "Cannot generate training data."
        )

    return config


def run_eval_splits(
    config: dict,
    volume_override: int | None = None,
    *,
    tiers: Sequence[int] | None = None,
    eval_split_volume: int | None = None,
    refresh_eval_splits: bool = False,
) -> frozenset[str]:
    """Generate eval splits, freeze benchmark, and return the blocklist."""
    selected_splits = set(eval_splits_for_tiers(list(tiers or [])))
    split_volume = eval_split_volume if eval_split_volume is not None else volume_override
    all_openings_count = len(config.get("openings", []))
    prepared_sources = build_eval_split_sources(
        config,
        min_depth_eval_benchmark=MIN_DEPTH_EVAL_BENCHMARK,
        seed=MASTER_SEED,
    )
    config["openings"] = prepared_sources.train_openings
    logger.info(
        "Openings ECO holdout: %d total -> %d eval, %d train",
        all_openings_count,
        len(prepared_sources.eval_openings),
        len(prepared_sources.train_openings),
    )

    blocklist_path = EVAL_SPLITS_DIR / "blocklist.txt"
    split_sizes = effective_eval_split_sizes(
        EVAL_SPLIT_SIZES,
        volume_override=split_volume,
    )
    split_sizes = {
        split_name: split_size
        for split_name, split_size in split_sizes.items()
        if split_name in selected_splits
    }
    selected_sources = {
        split_name: prepared_sources.sources[split_name]
        for split_name in split_sizes
    }
    split_sizes = reserve_training_rows_for_volume(
        split_sizes,
        prepared_sources,
        volume_override=split_volume,
    )
    expected_manifest = build_eval_split_manifest(
        prepared_sources,
        split_sizes=split_sizes,
        volume_override=volume_override,
        eval_split_volume=eval_split_volume,
        fingerprint_cache_path=EVAL_SPLITS_DIR / EVAL_SPLIT_FINGERPRINT_CACHE_NAME,
    )
    if blocklist_path.exists() and eval_split_manifest_matches(
        EVAL_SPLITS_DIR,
        expected_manifest,
    ):
        logger.info("Loading existing blocklist from %s", blocklist_path)
        blocklist = load_blocklist(blocklist_path)
        if not (BENCHMARK_DIR / "manifest.json").exists():
            logger.info("Benchmark missing; freezing from existing eval splits...")
            _freeze_from_disk(split_names=split_sizes)
        return blocklist
    existing_blocklist: frozenset[str] = frozenset()
    if blocklist_path.exists():
        if _has_existing_tier_outputs(tiers) and not refresh_eval_splits:
            raise RuntimeError(
                "eval split manifest changed while tier outputs already exist; "
                "use --refresh-eval-splits or a fresh output root"
            )
        logger.info(
            "Existing eval blocklist does not match current generation settings; "
            "rebuilding eval splits (merging with the existing blocklist)."
        )
        existing_blocklist = load_blocklist(blocklist_path)

    logger.info("Generating eval splits...")
    all_evals = config.get("position_evals", [])
    logger.info(
        "Evaluation split: %d / %d evals at depth >= %d",
        len(prepared_sources.eval_benchmark_evals),
        len(all_evals),
        MIN_DEPTH_EVAL_BENCHMARK,
    )

    fen_pool = config.get("fen_pool", [])
    splits = generate_all_eval_splits(selected_sources, split_sizes=split_sizes)
    save_eval_splits(
        splits,
        EVAL_SPLITS_DIR,
        fen_pool=fen_pool,
        extra_blocklist_keys=existing_blocklist,
    )
    write_eval_split_manifest(expected_manifest, EVAL_SPLITS_DIR)
    freeze_and_save(
        splits,
        BENCHMARK_DIR,
        seed=MASTER_SEED,
        clean=not _benchmark_has_other_split_files(split_sizes),
        strict_coverage=(volume_override is None and eval_split_volume is None),
    )
    return build_blocklist(splits, fen_pool) | existing_blocklist


def build_eval_split_manifest(
    prepared_sources,
    *,
    split_sizes: dict[str, int],
    volume_override: int | None,
    eval_split_volume: int | None = None,
    master_seed: int | None = None,
    min_depth_eval_benchmark: int | None = None,
    fingerprint_cache_path: str | Path | None = None,
) -> dict:
    """Build a deterministic manifest for deciding eval split reuse."""
    return {
        "artifact_type": "sft_eval_split_manifest",
        "schema_version": "1.0",
        "master_seed": MASTER_SEED if master_seed is None else master_seed,
        "volume_override": volume_override,
        "eval_split_volume": eval_split_volume,
        "min_depth_eval_benchmark": (
            MIN_DEPTH_EVAL_BENCHMARK
            if min_depth_eval_benchmark is None
            else min_depth_eval_benchmark
        ),
        "split_sizes": dict(sorted(split_sizes.items())),
        "source_counts": {
            name: len(rows)
            for name, rows in sorted(prepared_sources.sources.items())
        },
        "source_fingerprints": _source_fingerprints(
            prepared_sources.sources,
            cache_path=(
                Path(fingerprint_cache_path) if fingerprint_cache_path else None
            ),
        ),
    }


def eval_split_manifest_matches(path: str | Path, expected_manifest: dict) -> bool:
    """Return True when the on-disk eval split manifest matches this run."""
    manifest_path = Path(path) / EVAL_SPLIT_MANIFEST_NAME
    if not manifest_path.exists():
        return False
    try:
        with manifest_path.open(encoding="utf-8") as fh:
            actual = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected_manifest


def write_eval_split_manifest(manifest: dict, path: str | Path) -> Path:
    """Write the eval split manifest and return its path."""
    manifest_path = Path(path) / EVAL_SPLIT_MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest_path


def _source_fingerprint(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(raw_fen_identity_key(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _source_fingerprint_cache_key(rows: list[dict]) -> str:
    """Cheap content hash over the raw fields _source_fingerprint depends on."""
    digest = hashlib.sha256()
    for row in rows:
        if not isinstance(row, Mapping):
            digest.update(str(row).encode("utf-8"))
            digest.update(b"\n")
            continue
        digest.update(str(row.get("fen", "")).encode("utf-8"))
        digest.update(b"|1" if row.get("is_chess960") else b"|0")
        chess960_id = row.get("chess960_id")
        metadata = row.get("metadata")
        if isinstance(metadata, Mapping):
            if chess960_id is None:
                chess960_id = metadata.get("chess960_id")
            if metadata.get("is_chess960"):
                digest.update(b"|m")
        digest.update(str(chess960_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_fingerprint_cache(cache_path: Path) -> dict:
    try:
        with cache_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    entries = payload.get("sources") if isinstance(payload, dict) else None
    return entries if isinstance(entries, dict) else {}


def _source_fingerprints(
    sources: Mapping[str, list[dict]],
    *,
    cache_path: Path | None = None,
) -> dict[str, str]:
    """Fingerprint sources, reusing cached digests for unchanged row sets.

    ``_source_fingerprint`` parses every row with python-chess, which is slow
    at scale; the sidecar cache keyed on a cheap raw-content hash lets repeat
    invocations with unchanged sources skip the re-parse entirely.
    """
    cached = _load_fingerprint_cache(cache_path) if cache_path else {}
    fingerprints: dict[str, str] = {}
    updated: dict[str, dict[str, str]] = {}
    for name, rows in sorted(sources.items()):
        cache_key = _source_fingerprint_cache_key(rows)
        entry = cached.get(name)
        if (
            isinstance(entry, Mapping)
            and entry.get("cache_key") == cache_key
            and entry.get("fingerprint")
        ):
            fingerprints[name] = str(entry["fingerprint"])
        else:
            fingerprints[name] = _source_fingerprint(rows)
        updated[name] = {"cache_key": cache_key, "fingerprint": fingerprints[name]}
    if cache_path is not None and updated != cached:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8", newline="\n") as fh:
                json.dump(
                    {
                        "artifact_type": "sft_source_fingerprint_cache",
                        "schema_version": "1.0",
                        "sources": updated,
                    },
                    fh,
                    indent=2,
                    sort_keys=True,
                )
                fh.write("\n")
        except OSError:  # pragma: no cover - depends on filesystem state
            logger.warning("Could not write fingerprint cache to %s", cache_path)
    return fingerprints


def _benchmark_has_other_split_files(split_names: Sequence[str]) -> bool:
    """Return True when the benchmark dir holds splits outside *split_names*."""
    benchmark_dir = Path(BENCHMARK_DIR)
    if not benchmark_dir.exists():
        return False
    keep = {f"{split_name}.jsonl" for split_name in split_names}
    return any(path.name not in keep for path in benchmark_dir.glob("*.jsonl"))


def _freeze_from_disk(*, split_names: Sequence[str] | None = None) -> None:
    """Re-freeze benchmark from existing eval split JSONL on disk."""
    splits: dict[str, list[dict]] = {}
    for split_name in split_names or EVAL_SPLIT_SIZES:
        path = EVAL_SPLITS_DIR / f"{split_name}.jsonl"
        if not path.exists():
            continue
        rows: list[dict] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        splits[split_name] = rows

    if splits:
        freeze_and_save(
            splits,
            BENCHMARK_DIR,
            seed=MASTER_SEED,
            strict_coverage=False,
        )


def _scan_task_output(
    output_path: Path,
    task_id: str,
    stats: PipelineStats | None = None,
    blocklist: frozenset[str] | None = None,
) -> tuple[int, int]:
    """Return non-empty line count and validation error count for a task file."""
    count = 0
    errors = 0
    with output_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                if stats is not None:
                    stats.record(task_id, passed_validation=False)
                continue

            passed, _ = validate_example(obj)
            if obj.get("task") != task_id:
                passed = False
            if passed and blocklist is not None:
                passed = check_row_no_contamination(obj, blocklist)
            if not passed:
                errors += 1
            if stats is not None:
                stats.record(
                    task_id,
                    is_chess960=obj.get("is_chess960", False),
                    passed_validation=passed,
                )

    return count, errors


def _task_manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(".manifest.json")


def _example_identity(example: dict) -> str | None:
    metadata = example.get("metadata")
    if not isinstance(metadata, dict):
        return None
    identity = metadata.get("example_identity")
    return str(identity) if identity else None


def _scan_task_output_identities(
    output_path: Path,
    task_id: str,
    blocklist: frozenset[str] | None = None,
) -> tuple[int, int, set[str], bool]:
    """Return count, error count, identities, and whether any valid row lacks one."""
    count = 0
    errors = 0
    identities: set[str] = set()
    missing_identity = False
    with output_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            passed, _ = validate_example(obj)
            if obj.get("task") != task_id:
                passed = False
            if passed and blocklist is not None:
                passed = check_row_no_contamination(obj, blocklist)
            if not passed:
                errors += 1
                continue
            identity = _example_identity(obj)
            if identity is None:
                missing_identity = True
            elif identity in identities:
                errors += 1
            else:
                identities.add(identity)
    return count, errors, identities, missing_identity


def _generation_source_fingerprint(config: dict) -> str:
    digest = hashlib.sha256()
    for key in (
        "fen_pool",
        "game_positions",
        "puzzles",
        "openings",
        "position_evals",
        "best_move_evals",
        "candidate_rating_evals",
        "endgame_positions",
        "mate_rows",
    ):
        rows = config.get(key, [])
        digest.update(key.encode("utf-8"))
        digest.update(str(len(rows)).encode("utf-8"))
        digest.update(b"\n")
        for row in rows[:1000]:
            if isinstance(row, Mapping):
                digest.update(raw_fen_identity_key(row).encode("utf-8"))
            else:
                digest.update(str(row).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def _write_task_generation_manifest(
    output_path: Path,
    *,
    task_id: str,
    target_count: int,
    previous_count: int,
    final_count: int,
    appended_count: int,
    skipped_duplicate_count: int,
    source_fingerprint: str,
) -> None:
    payload = {
        "artifact_type": "sft_task_generation_manifest",
        "schema_version": "1.0",
        "task_id": task_id,
        "target_count": target_count,
        "previous_count": previous_count,
        "final_count": final_count,
        "appended_count": appended_count,
        "skipped_duplicate_count": skipped_duplicate_count,
        "source_fingerprint": source_fingerprint,
    }
    manifest_path = _task_manifest_path(output_path)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _extend_task_output(
    output_path: Path,
    gen_cls: type,
    gen_config: dict,
    blocklist: frozenset[str],
    *,
    task_id: str,
    target: int,
    existing_count: int,
    existing_identities: set[str],
) -> tuple[int, int, int]:
    """Append unseen generated rows to an existing valid task file."""
    candidate_config = dict(gen_config)
    if candidate_config.get("volume_override") is not None:
        candidate_config["volume_override"] = target + existing_count
    gen = gen_cls(config=candidate_config, blocklist=blocklist, rng=_task_rng(task_id))
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    appended = 0
    skipped_duplicate = 0
    errors = 0
    identities = set(existing_identities)

    try:
        with output_path.open(encoding="utf-8") as src, tmp_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as dst:
            for line in src:
                dst.write(line)
            final_count = existing_count
            for example in gen.generate():
                identity = _example_identity(example)
                if identity is None:
                    errors += 1
                    continue
                if identity in identities:
                    skipped_duplicate += 1
                    continue
                passed, _ = validate_example(example)
                if example.get("task") != task_id:
                    passed = False
                if passed:
                    passed = check_row_no_contamination(example, blocklist)
                if not passed:
                    errors += 1
                    continue
                dst.write(json.dumps(example, ensure_ascii=False) + "\n")
                identities.add(identity)
                appended += 1
                final_count += 1
                if final_count >= target:
                    break
        if errors:
            raise RuntimeError(f"{task_id} has {errors} validation error(s) while extending")
        if final_count < target:
            raise RuntimeError(f"{task_id} underfilled: wrote {final_count} / {target}")
        tmp_path.replace(output_path)
        return final_count, appended, skipped_duplicate
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _task_rng(task_id: str) -> Random:
    """Return a per-task RNG that is stable regardless of task run order."""
    return Random(f"{MASTER_SEED}:{task_id}")


def _assert_task_complete(task_id: str, count: int, errors: int, target: int) -> None:
    """Fail generation when the task file is invalid or below target volume."""
    if errors:
        raise RuntimeError(f"{task_id} has {errors} validation error(s)")
    if count < target:
        raise RuntimeError(f"{task_id} underfilled: wrote {count} / {target}")


def run_tier(
    tier: int,
    config: dict,
    blocklist: frozenset[str],
    stats: PipelineStats,
    volume_override: int | None = None,
) -> None:
    """Run all generators for a single tier."""
    generators = TIER_GENERATORS.get(tier, [])
    if not generators:
        logger.warning("No generators for tier %d", tier)
        return

    tier_dir = TIER_OUTPUT_DIR / f"tier{tier}"
    tier_dir.mkdir(parents=True, exist_ok=True)

    gen_config = dict(config)
    if volume_override is not None:
        gen_config["volume_override"] = volume_override
    source_fingerprint = _generation_source_fingerprint(gen_config)

    for gen_cls in generators:
        task_id = gen_cls(config=gen_config, blocklist=blocklist).task_id()
        # A per-task RNG keeps generated data independent of which sibling
        # tasks were skipped or resumed in this invocation.
        gen = gen_cls(config=gen_config, blocklist=blocklist, rng=_task_rng(task_id))
        target = gen.target_volume()
        output_path = tier_dir / f"{task_id}.jsonl"

        if output_path.exists() and output_path.stat().st_size > 0:
            existing, existing_errors, existing_identities, missing_identity = (
                _scan_task_output_identities(
                    output_path,
                    task_id,
                    blocklist=blocklist,
                )
            )
            if existing_errors == 0 and existing >= target:
                _scan_task_output(output_path, task_id, stats, blocklist=blocklist)
                _write_task_generation_manifest(
                    output_path,
                    task_id=task_id,
                    target_count=target,
                    previous_count=existing,
                    final_count=existing,
                    appended_count=0,
                    skipped_duplicate_count=0,
                    source_fingerprint=source_fingerprint,
                )
                logger.info(
                    "Skipping %s - output exists at %s (%d examples)",
                    task_id,
                    output_path,
                    existing,
                )
                continue
            if existing_errors == 0 and not missing_identity and existing < target:
                logger.info(
                    "Extending %s - existing output has %d / %d examples",
                    task_id,
                    existing,
                    target,
                )
                final_count, appended_count, skipped_duplicate_count = _extend_task_output(
                    output_path,
                    gen_cls,
                    gen_config,
                    blocklist,
                    task_id=task_id,
                    target=target,
                    existing_count=existing,
                    existing_identities=existing_identities,
                )
                _scan_task_output(output_path, task_id, stats, blocklist=blocklist)
                _write_task_generation_manifest(
                    output_path,
                    task_id=task_id,
                    target_count=target,
                    previous_count=existing,
                    final_count=final_count,
                    appended_count=appended_count,
                    skipped_duplicate_count=skipped_duplicate_count,
                    source_fingerprint=source_fingerprint,
                )
                logger.info(
                    "  %s: %d final, %d appended, %d duplicate candidate(s) skipped",
                    task_id,
                    final_count,
                    appended_count,
                    skipped_duplicate_count,
                )
                continue
            existing_count, existing_errors = _scan_task_output(
                output_path,
                task_id,
                blocklist=blocklist,
            )
            logger.info(
                "Regenerating %s - existing output incomplete or invalid "
                "(%d / %d examples, %d error(s))",
                task_id,
                existing_count,
                target,
                existing_errors,
            )

        logger.info("Generating %s (target: %d)...", task_id, target)
        with JSONLWriter(
            output_path,
            commit_on_close=False,
            expected_task_id=task_id,
        ) as writer:
            for example in gen.generate():
                written = writer.write(example)
                stats.record(
                    task_id,
                    is_chess960=example.get("is_chess960", False),
                    passed_validation=written,
                )
            _assert_task_complete(task_id, writer.count, writer.error_count, target)
            writer.commit()
            _write_task_generation_manifest(
                output_path,
                task_id=task_id,
                target_count=target,
                previous_count=0,
                final_count=writer.count,
                appended_count=writer.count,
                skipped_duplicate_count=0,
                source_fingerprint=source_fingerprint,
            )

        logger.info(
            "  %s: %d written, %d errors",
            task_id,
            writer.count,
            writer.error_count,
        )


def validate_existing_outputs(
    volume_override: int | None = None,
    *,
    tiers: Sequence[int] | None = None,
) -> int:
    """Validate existing tier outputs, completeness, and decontamination."""
    blocklist_path = EVAL_SPLITS_DIR / "blocklist.txt"
    if not blocklist_path.exists():
        logger.error("No blocklist found at %s; run eval splits first", blocklist_path)
        return 1

    total_examples = 0
    total_errors = 0
    for jsonl_path in iter_jsonl_artifacts(TIER_OUTPUT_DIR):
        if tiers is not None and not _output_file_matches_tiers(jsonl_path, tiers):
            continue
        file_count = 0
        file_errors = 0
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                except json.JSONDecodeError:
                    file_errors += 1
                    continue
                file_count += 1
                passed, _ = validate_example(example)
                if not passed:
                    file_errors += 1
        total_examples += file_count
        total_errors += file_errors
        if file_errors:
            logger.warning(
                "%s: %d / %d examples failed validation",
                jsonl_path.name,
                file_errors,
                file_count,
            )

    logger.info(
        "Validation: %d examples, %d errors (%.3f%%)",
        total_examples,
        total_errors,
        total_errors / total_examples * 100 if total_examples else 0,
    )

    completeness = audit_output_completeness(
        TIER_OUTPUT_DIR,
        expected_volume_override=volume_override,
        tiers=tiers,
    )
    if completeness:
        logger.error("Completeness issues found:")
        for rel_path, issue in completeness.items():
            logger.error("  %s: %s", rel_path, issue)
        total_errors += len(completeness)
    else:
        logger.info("Completeness check passed.")

    blocklist = load_blocklist(blocklist_path)
    report = audit_output_files(TIER_OUTPUT_DIR, blocklist)
    if report:
        logger.error("Contamination found:")
        for file_path, fens in report.items():
            logger.error("  %s: %d contaminated FENs", file_path, len(fens))
        total_errors += len(report)
    else:
        logger.info("No contamination found. All outputs clean.")

    return 1 if total_errors > 0 else 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chess SFT data generation pipeline")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--tier", type=int, nargs="+", help="Run specific tier(s)")
    parser.add_argument("--eval-only", action="store_true", help="Only generate eval splits")
    parser.add_argument("--volume", type=int, help="Override volume per task for testing")
    parser.add_argument(
        "--eval-split-volume",
        type=int,
        help=(
            "Reserve eval splits as if this per-task volume were used; useful "
            "when extending a small data root toward a larger run."
        ),
    )
    parser.add_argument(
        "--refresh-eval-splits",
        action="store_true",
        help="Allow rebuilding eval splits even when tier outputs already exist.",
    )
    parser.add_argument("--validate-only", action="store_true", help="Re-validate existing outputs")
    parser.add_argument(
        "--allow-source-gaps",
        action="store_true",
        help="Warn about missing required source families instead of failing before generation",
    )
    parser.add_argument(
        "--source-readiness-report",
        type=Path,
        help="Write a JSON manifest with loaded source counts and missing source families",
    )
    return parser


def _output_file_matches_tiers(jsonl_path: Path, tiers: Sequence[int]) -> bool:
    selected = {f"tier{tier}" for tier in tiers}
    try:
        relative = jsonl_path.relative_to(TIER_OUTPUT_DIR)
    except ValueError:
        relative = jsonl_path
    return bool(relative.parts) and relative.parts[0] in selected


def _has_existing_tier_outputs(tiers: Sequence[int] | None = None) -> bool:
    for jsonl_path in iter_jsonl_artifacts(TIER_OUTPUT_DIR):
        if tiers is not None and not _output_file_matches_tiers(jsonl_path, tiers):
            continue
        return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not (args.validate_only or args.eval_only or args.all or args.tier):
        parser.print_help()
        return 0

    _ensure_output_dirs()

    if args.validate_only:
        return validate_existing_outputs(
            volume_override=args.volume,
            tiers=args.tier,
        )

    stats = _pipeline_stats_for_volume_override(args.volume)
    config = load_sources(volume_override=args.volume)

    if args.eval_only:
        tiers = args.tier or []
    elif args.all:
        tiers = list(range(1, 8))
    else:
        tiers = args.tier or []

    blocklist_path = EVAL_SPLITS_DIR / "blocklist.txt"
    strict_eval_splits = not blocklist_path.exists() and args.volume is None
    readiness = build_source_readiness_report(
        config,
        tiers=tiers,
        strict_eval_splits=strict_eval_splits,
    )
    for line in readiness.format_lines():
        log = logger.info if readiness.ok else logger.warning
        log(line)
    if args.source_readiness_report:
        write_source_readiness_report(readiness, args.source_readiness_report)

    if readiness.counts.get("fen_pool", 0) <= 0:
        logger.error("All sources failed; fen_pool is empty. Aborting.")
        return 1
    if not readiness.ok and not args.allow_source_gaps and args.volume is None:
        logger.error(
            "Source readiness failed. Use --allow-source-gaps to continue anyway "
            "or --volume for a permissive smoke run."
        )
        return 1

    eval_split_kwargs = {
        "volume_override": args.volume,
        "tiers": tiers,
    }
    if args.eval_split_volume is not None:
        eval_split_kwargs["eval_split_volume"] = args.eval_split_volume
    if args.refresh_eval_splits:
        eval_split_kwargs["refresh_eval_splits"] = args.refresh_eval_splits
    blocklist = run_eval_splits(config, **eval_split_kwargs)
    logger.info("Eval blocklist: %d FENs", len(blocklist))

    if args.eval_only:
        return 0

    for tier in sorted(tiers):
        logger.info("=== Tier %d ===", tier)
        run_tier(tier, config, blocklist, stats, volume_override=args.volume)

    print("\n" + stats.report())
    mix = stats.verify_chess960_mix()
    print("\nChess960 Mix Verification:")
    for tier_key, info in sorted(mix.items()):
        status = "OK" if abs(info["delta"]) < 0.02 else "DRIFT"
        print(
            f"  {tier_key}: target={info['target']:.0%} "
            f"actual={info['actual']:.0%} [{status}]"
        )

    return 0


def cli(argv: Sequence[str] | None = None) -> None:
    """Console-script wrapper that avoids third-party finalizer aborts."""
    exit_code = main(argv)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def _ensure_output_dirs() -> None:
    for directory in (
        OUTPUT_DIR,
        POOL_DIR,
        EVAL_SPLITS_DIR,
        ANNOTATIONS_DIR,
        TIER_OUTPUT_DIR,
        BENCHMARK_DIR,
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)


__all__ = [
    "ANNOTATIONS_DIR",
    "BENCHMARK_DIR",
    "EVAL_SPLITS_DIR",
    "MASTER_SEED",
    "OUTPUT_DIR",
    "POOL_DIR",
    "SETTINGS",
    "TIER_GENERATORS",
    "TIER_OUTPUT_DIR",
    "build_eval_split_manifest",
    "build_arg_parser",
    "cli",
    "eval_split_manifest_matches",
    "chess960_source_target_count",
    "load_sources",
    "main",
    "run_eval_splits",
    "run_tier",
    "_output_file_matches_tiers",
    "write_eval_split_manifest",
    "validate_existing_outputs",
]


if __name__ == "__main__":
    cli()
