#!/usr/bin/env python3
"""Main CLI: orchestrate the full SFT data generation pipeline.

Usage:
    python run_pipeline.py --all              # full pipeline
    python run_pipeline.py --tier 1           # just Tier 1
    python run_pipeline.py --tier 1 2         # Tiers 1 and 2
    python run_pipeline.py --eval-only        # just eval splits
    python run_pipeline.py --volume 100       # tiny test run (100 per task)
    python run_pipeline.py --validate-only    # re-validate existing outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from random import Random

# Ensure make_data is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    MASTER_SEED,
    OUTPUT_DIR,
    EVAL_SPLITS_DIR,
    TIER_OUTPUT_DIR,
    POOL_DIR,
    ANNOTATIONS_DIR,
    BENCHMARK_DIR,
    POLYGLOT_DIR,
    SYZYGY_PATH,
)
from pool.eval_split import (
    build_blocklist,
    generate_all_eval_splits,
    load_blocklist,
    partition_eco_codes,
    save_eval_splits,
)
from config.settings import MIN_DEPTH_EVAL_BENCHMARK
from validation.benchmark import freeze_and_save
from pool.fen_pool import FENPool
from pool.annotator import BatchAnnotator
from output.writer import JSONLWriter
from output.stats import PipelineStats
from validation.decontamination import audit_output_files
from validation.completeness import audit_output_completeness

# Generator imports
from generators.tier1_perception import (
    FENToBoard, BoardToFEN, PieceIdentification, PieceCounting, StateTracking,
)
from generators.tier2_rules import (
    LegalMoveGen, PieceSpecificMoves, MoveLegalityCheck, CheckDetection, SpecialRules,
)
from generators.tier3_tactics import (
    AvailableCaptures, Threats, AttackedDefended, TacticalPatterns, HangingPieces,
)
from generators.tier4_evaluation import (
    MaterialBalance, PositionEvaluation, PawnStructure,
)
from generators.tier5_openings import (
    OpeningIdentification, OpeningContinuation, OpeningPrinciples,
)
from generators.tier6_endgames import (
    EndgameClassification, EndgameWDL, EndgameBestMove, EndgamePrinciples,
)
from generators.tier7_planning import (
    BestMoveSelection, PuzzleSolving, MoveConsequence,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# All generator classes grouped by tier
TIER_GENERATORS: dict[int, list[type]] = {
    1: [FENToBoard, BoardToFEN, PieceIdentification, PieceCounting, StateTracking],
    2: [LegalMoveGen, PieceSpecificMoves, MoveLegalityCheck, CheckDetection, SpecialRules],
    3: [AvailableCaptures, Threats, AttackedDefended, TacticalPatterns, HangingPieces],
    4: [MaterialBalance, PositionEvaluation, PawnStructure],
    5: [OpeningIdentification, OpeningContinuation, OpeningPrinciples],
    6: [EndgameClassification, EndgameWDL, EndgameBestMove, EndgamePrinciples],
    7: [BestMoveSelection, PuzzleSolving, MoveConsequence],
}


def load_sources(volume_override: int | None = None) -> dict:
    """Load all data sources and build the shared config dict.

    This populates the FEN pool, puzzles, openings, evals, endgame
    positions, etc. that generators consume via ``self.config``.

    Memory caps are applied even for full runs (``volume_override=None``)
    so the pipeline doesn't OOM on real data volumes.
    """
    logger.info("Loading data sources...")
    rng = Random(MASTER_SEED)

    if volume_override:
        max_items = volume_override * 10
    else:
        # Sensible caps for full runs — enough to fill every generator's
        # target volume with headroom, without holding hundreds of millions
        # of rows in memory.
        max_items = 500_000

    config: dict = {}

    # --- Source 1: Lichess Games -> FEN pool ---
    logger.info("Loading Lichess games...")
    from sources.lichess_games import stream_games, extract_positions

    fen_pool_entries: list[dict] = []
    game_positions: list[dict] = []
    try:
        for game in stream_games(max_games=max_items):
            for pos in extract_positions(game):
                entry = {"fen": pos["fen"], "source": "lichess_games",
                         "game_phase": pos["game_phase"], "is_chess960": False}
                fen_pool_entries.append(entry)
                game_positions.append(pos)
                if len(fen_pool_entries) >= max_items:
                    break
            if len(fen_pool_entries) >= max_items:
                break
    except Exception as exc:
        logger.warning("Lichess games loading failed: %s", exc)

    logger.info("Lichess games: %d FENs, %d game_positions", len(fen_pool_entries), len(game_positions))

    # --- Source 2: Lichess Puzzles ---
    logger.info("Loading Lichess puzzles...")
    from sources.lichess_puzzles import load_puzzles

    puzzles: list[dict] = []
    try:
        for puzzle in load_puzzles(max_puzzles=max_items):
            puzzles.append(puzzle)
    except Exception as exc:
        logger.warning("Lichess puzzles loading failed: %s", exc)

    # --- Source 3: Lichess Openings ---
    logger.info("Loading Lichess openings...")
    from sources.lichess_openings import load_openings

    openings: list[dict] = []
    try:
        for opening in load_openings(max_openings=max_items):
            openings.append(opening)
            fen_pool_entries.append({"fen": opening["fen"], "source": "lichess_openings",
                                     "is_chess960": False})
    except Exception as exc:
        logger.warning("Lichess openings loading failed: %s", exc)

    # --- Source 4: Lichess Evals ---
    logger.info("Loading Lichess position evaluations...")
    from sources.lichess_evals import stream_evals

    position_evals: list[dict] = []
    best_move_evals: list[dict] = []
    try:
        for ev in stream_evals(min_depth=20, max_rows=max_items):
            position_evals.append(ev)
            if ev.get("depth", 0) >= 30:
                best_move_evals.append(ev)
    except Exception as exc:
        logger.warning("Lichess evals loading failed: %s", exc)

    # Add lichess eval FENs to the pool — these are real game positions
    # with depth-20+ Stockfish analysis, already validated.
    for ev in position_evals:
        fen_pool_entries.append({
            "fen": ev["fen"], "source": "lichess_evals",
            "game_phase": "unknown", "is_chess960": False,
        })
    logger.info("Added %d lichess eval FENs to pool (total: %d)", len(position_evals), len(fen_pool_entries))

    # Supplement game_positions from evals if games yielded too few
    if len(game_positions) < 50_000:
        supplement_count = 0
        for ev in position_evals:
            if ev.get("fen") and ev.get("best_move"):
                game_positions.append({
                    "fen": ev["fen"],
                    "move_played_uci": ev["best_move"],
                    "game_phase": "unknown",
                    "material_balance": 0,
                    "ply": 0,
                })
                supplement_count += 1
        logger.info(
            "Supplemented game_positions with %d eval entries (total: %d)",
            supplement_count, len(game_positions),
        )

    # --- Source 5: Polyglot books ---
    logger.info("Loading Polyglot opening books...")
    from sources.polyglot_books import load_book, get_weighted_moves

    import chess
    book_moves: dict[str, dict[str, int]] = {}  # fen -> {uci: total_weight}
    polyglot_dir = Path(POLYGLOT_DIR)
    if polyglot_dir.exists():
        for bin_file in polyglot_dir.glob("*.bin"):
            try:
                with load_book(str(bin_file)) as reader:
                    for opening in openings:
                        board = chess.Board(opening["fen"])
                        moves = get_weighted_moves(reader, board)
                        if moves:
                            merged = book_moves.setdefault(opening["fen"], {})
                            for uci, weight in moves:
                                merged[uci] = merged.get(uci, 0) + weight
            except Exception as exc:
                logger.warning("Polyglot book %s failed: %s", bin_file, exc)

    # Convert merged dicts to sorted (uci, weight) lists
    book_moves_list: dict[str, list] = {}
    for fen, move_weights in book_moves.items():
        book_moves_list[fen] = sorted(
            move_weights.items(), key=lambda x: x[1], reverse=True,
        )
    book_moves = book_moves_list  # type: ignore[assignment]

    # --- Source 6: Syzygy (loaded on demand by generators) ---
    endgame_positions: list[dict] = []
    syzygy_path = Path(SYZYGY_PATH)
    if syzygy_path.exists():
        logger.info("Loading Syzygy endgame positions...")
        from sources.syzygy_probing import open_tablebase, sample_endgame_positions, MATERIAL_CONFIGS, best_dtz_move

        try:
            with open_tablebase(str(syzygy_path)) as tb:
                for mat in MATERIAL_CONFIGS:
                    n = volume_override or 3000
                    for pos in sample_endgame_positions(tb, mat, n, rng):
                        board = chess.Board(pos["fen"])
                        bm = best_dtz_move(tb, board)
                        pos["best_move"] = bm or ""
                        endgame_positions.append(pos)
        except Exception as exc:
            logger.warning("Syzygy loading failed: %s", exc)

    # --- Source 8: Chess960 positions ---
    logger.info("Generating Chess960 positions...")
    from sources.chess960 import sample_chess960_positions

    # Size Chess960 dynamically to hit ~15% of total pool
    if volume_override:
        n_chess960 = volume_override
    else:
        n_standard = len(fen_pool_entries)
        target_960_ratio = 0.15
        n_chess960 = int(n_standard * target_960_ratio / (1 - target_960_ratio))
        n_chess960 = max(5_000, min(n_chess960, 100_000))
    logger.info("Chess960 target: %d (standard pool: %d)", n_chess960, len(fen_pool_entries))

    chess960_positions = sample_chess960_positions(
        n=n_chess960, n_random_moves=20, rng=rng
    )
    for pos in chess960_positions:
        fen_pool_entries.append(pos)

    # --- Source 9: MATE dataset ---
    logger.info("Loading MATE dataset...")
    from sources.mate_dataset import load_mate

    mate_rows: list[dict] = []
    try:
        for row in load_mate(max_rows=max_items):
            mate_rows.append(row)
    except Exception as exc:
        logger.warning("MATE dataset loading failed: %s", exc)

    # Deduplicate via FENPool, then convert back to list
    logger.info("Deduplicating FEN pool via FENPool (%d raw entries)...", len(fen_pool_entries))
    pool = FENPool()
    for entry in fen_pool_entries:
        fen = entry["fen"] if isinstance(entry, dict) else entry
        tags = entry if isinstance(entry, dict) else {"fen": fen}
        pool.add(fen, **{k: v for k, v in tags.items() if k != "fen"})
    logger.info("FEN pool after dedup: %d unique FENs", len(pool))

    # Convert back to list and shuffle for variety
    fen_pool_entries = [{"fen": f, **pool.get_tags(f)} for f in pool.all_fens()]
    rng.shuffle(fen_pool_entries)

    config["fen_pool"] = fen_pool_entries
    config["game_positions"] = game_positions
    config["puzzles"] = puzzles
    config["openings"] = openings
    config["position_evals"] = position_evals
    config["best_move_evals"] = best_move_evals
    config["consequence_evals"] = position_evals  # reuse for move consequence
    config["book_moves"] = book_moves
    config["endgame_positions"] = endgame_positions
    config["mate_rows"] = mate_rows

    logger.info(
        "Sources loaded: %d FENs, %d puzzles, %d openings, "
        "%d evals, %d endgames, %d MATE rows",
        len(fen_pool_entries), len(puzzles), len(openings),
        len(position_evals), len(endgame_positions), len(mate_rows),
    )

    if not fen_pool_entries:
        logger.error(
            "CRITICAL: fen_pool is empty after loading all sources. "
            "Cannot generate training data."
        )

    return config


def run_eval_splits(config: dict, volume_override: int | None = None) -> frozenset[str]:
    """Generate eval splits, freeze benchmark, and return the blocklist.

    On reruns (blocklist already exists) this still performs ECO holdout
    so that ``config["openings"]`` is trimmed to train-only rows, and
    produces the frozen benchmark if it is missing.
    """
    # ECO-based holdout ALWAYS runs so config["openings"] is train-only.
    all_openings = config.get("openings", [])
    if all_openings:
        eval_openings, train_openings = partition_eco_codes(all_openings)
    else:
        eval_openings, train_openings = [], []
    config["openings"] = train_openings
    logger.info(
        "Openings ECO holdout: %d total -> %d eval, %d train",
        len(all_openings), len(eval_openings), len(train_openings),
    )

    blocklist_path = EVAL_SPLITS_DIR / "blocklist.txt"

    if blocklist_path.exists():
        logger.info("Loading existing blocklist from %s", blocklist_path)
        blocklist = load_blocklist(str(blocklist_path))
        # Ensure frozen benchmark exists even on reruns
        if not (BENCHMARK_DIR / "manifest.json").exists():
            logger.info("Benchmark missing — freezing from existing eval splits...")
            _freeze_from_disk()
        return blocklist

    logger.info("Generating eval splits...")

    # Enrich eval openings with book moves data for continuation benchmarking
    book_moves = config.get("book_moves", {})
    for opening in eval_openings:
        fen = opening.get("fen", "")
        if fen in book_moves:
            opening["book_moves"] = book_moves[fen]

    # Evaluation split: require depth >= 40 for higher quality
    all_evals = config.get("position_evals", [])
    eval_benchmark_evals = [
        e for e in all_evals
        if e.get("depth", 0) >= MIN_DEPTH_EVAL_BENCHMARK
    ]
    logger.info(
        "Evaluation split: %d / %d evals at depth >= %d",
        len(eval_benchmark_evals), len(all_evals), MIN_DEPTH_EVAL_BENCHMARK,
    )

    sources = {
        "perception": config.get("fen_pool", []),
        "rules": config.get("fen_pool", []),
        "tactics": config.get("puzzles", []),
        "evaluation": eval_benchmark_evals,
        "openings": eval_openings,
        "endgames": config.get("endgame_positions", []),
        "planning": config.get("puzzles", []) + config.get("best_move_evals", []),
        "chess960": [e for e in config.get("fen_pool", []) if e.get("is_chess960")],
        "mate": config.get("mate_rows", []),
    }

    splits = generate_all_eval_splits(sources)
    save_eval_splits(splits, str(EVAL_SPLITS_DIR))

    # Freeze benchmark from raw eval splits.
    # Relax coverage check for small test runs where some splits have
    # too few source rows to populate all task types.
    freeze_and_save(
        splits, str(BENCHMARK_DIR), seed=MASTER_SEED,
        strict_coverage=(volume_override is None),
    )

    return build_blocklist(splits)


def _freeze_from_disk() -> None:
    """Re-freeze benchmark from existing eval split JSONL on disk."""
    import json as _json
    from config.settings import EVAL_SPLIT_SIZES

    splits: dict[str, list[dict]] = {}
    for split_name in EVAL_SPLIT_SIZES:
        path = EVAL_SPLITS_DIR / f"{split_name}.jsonl"
        if not path.exists():
            continue
        rows: list[dict] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(_json.loads(line))
        splits[split_name] = rows

    if splits:
        freeze_and_save(
            splits, str(BENCHMARK_DIR), seed=MASTER_SEED,
            strict_coverage=False,
        )


def _scan_task_output(
    output_path: Path,
    task_id: str,
    stats: PipelineStats | None = None,
) -> tuple[int, int]:
    """Return non-empty line count and validation error count for one task file."""
    from validation.validator import validate_example as _val_ex

    count = 0
    errors = 0
    with open(output_path, encoding="utf-8") as fh:
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

            passed, _ = _val_ex(obj)
            if obj.get("task") != task_id:
                passed = False
            if not passed:
                errors += 1
            if stats is not None:
                stats.record(
                    task_id,
                    is_chess960=obj.get("is_chess960", False),
                    passed_validation=passed,
                )

    return count, errors


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

    rng = Random(MASTER_SEED + tier)
    tier_dir = TIER_OUTPUT_DIR / f"tier{tier}"
    tier_dir.mkdir(parents=True, exist_ok=True)

    gen_config = dict(config)
    if volume_override is not None:
        gen_config["volume_override"] = volume_override

    for gen_cls in generators:
        gen = gen_cls(config=gen_config, blocklist=blocklist, rng=rng)
        task_id = gen.task_id()
        target = gen.target_volume()
        output_path = tier_dir / f"{task_id}.jsonl"

        if output_path.exists() and output_path.stat().st_size > 0:
            existing, existing_errors = _scan_task_output(output_path, task_id)
            if existing_errors == 0 and existing >= target:
                _scan_task_output(output_path, task_id, stats)
                logger.info(
                    "Skipping %s - output exists at %s (%d examples)",
                    task_id, output_path, existing,
                )
                continue
            logger.info(
                "Regenerating %s - existing output incomplete or invalid "
                "(%d / %d examples, %d error(s))",
                task_id, existing, target, existing_errors,
            )

        logger.info("Generating %s (target: %d)...", task_id, target)

        with JSONLWriter(
            str(output_path),
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

        logger.info(
            "  %s: %d written, %d errors",
            task_id, writer.count, writer.error_count,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chess SFT data generation pipeline"
    )
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--tier", type=int, nargs="+", help="Run specific tier(s)")
    parser.add_argument("--eval-only", action="store_true", help="Only generate eval splits")
    parser.add_argument("--volume", type=int, help="Override volume per task (for testing)")
    parser.add_argument("--validate-only", action="store_true", help="Re-validate existing outputs")
    args = parser.parse_args()

    # Ensure output dirs exist
    for d in (OUTPUT_DIR, POOL_DIR, EVAL_SPLITS_DIR, ANNOTATIONS_DIR,
              TIER_OUTPUT_DIR, BENCHMARK_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)

    stats = PipelineStats()

    if args.validate_only:
        from validation.validator import validate_example

        blocklist_path = EVAL_SPLITS_DIR / "blocklist.txt"
        if not blocklist_path.exists():
            logger.error("No blocklist found at %s — run eval splits first", blocklist_path)
            return 1

        # 1. Per-example schema/semantic validation
        total_examples = 0
        total_errors = 0
        for jsonl_path in sorted(Path(TIER_OUTPUT_DIR).rglob("*.jsonl")):
            file_count = 0
            file_errors = 0
            with open(jsonl_path, encoding="utf-8") as fh:
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
                    passed, errs = validate_example(example)
                    if not passed:
                        file_errors += 1
            total_examples += file_count
            total_errors += file_errors
            if file_errors:
                logger.warning(
                    "%s: %d / %d examples failed validation",
                    jsonl_path.name, file_errors, file_count,
                )

        logger.info(
            "Validation: %d examples, %d errors (%.3f%%)",
            total_examples, total_errors,
            total_errors / total_examples * 100 if total_examples else 0,
        )

        # 2. Completeness audit
        completeness = audit_output_completeness(
            str(TIER_OUTPUT_DIR),
            expected_volume_override=args.volume,
        )
        if completeness:
            logger.error("Completeness issues found:")
            for rel_path, issue in completeness.items():
                logger.error("  %s: %s", rel_path, issue)
            total_errors += len(completeness)
        else:
            logger.info("Completeness check passed.")

        # 3. Decontamination audit
        blocklist = load_blocklist(str(blocklist_path))
        report = audit_output_files(str(TIER_OUTPUT_DIR), blocklist)
        if report:
            logger.error("Contamination found:")
            for f, fens in report.items():
                logger.error("  %s: %d contaminated FENs", f, len(fens))
            total_errors += len(report)
        logger.info("No contamination found. All outputs clean.")

        return 1 if total_errors > 0 else 0

    # Load sources
    config = load_sources(volume_override=args.volume)

    if not config.get("fen_pool"):
        logger.error("All sources failed — fen_pool is empty. Aborting.")
        return 1

    # Always run eval splits first
    blocklist = run_eval_splits(config, volume_override=args.volume)
    logger.info("Eval blocklist: %d FENs", len(blocklist))

    if args.eval_only:
        return 0

    # Determine which tiers to run
    if args.all:
        tiers = list(range(1, 8))
    elif args.tier:
        tiers = args.tier
    else:
        parser.print_help()
        return 0

    for tier in sorted(tiers):
        logger.info("=== Tier %d ===", tier)
        run_tier(tier, config, blocklist, stats, volume_override=args.volume)

    # Report
    print("\n" + stats.report())
    mix = stats.verify_chess960_mix()
    print("\nChess960 Mix Verification:")
    for tier_key, info in sorted(mix.items()):
        status = "OK" if abs(info["delta"]) < 0.02 else "DRIFT"
        print(f"  {tier_key}: target={info['target']:.0%} actual={info['actual']:.0%} [{status}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
