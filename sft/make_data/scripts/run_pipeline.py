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
    POLYGLOT_DIR,
    SYZYGY_PATH,
)
from pool.eval_split import (
    build_blocklist,
    generate_all_eval_splits,
    load_blocklist,
    save_eval_splits,
)
from pool.fen_pool import FENPool
from pool.annotator import BatchAnnotator
from output.writer import JSONLWriter
from output.stats import PipelineStats
from validation.decontamination import audit_output_files

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

    # --- Source 5: Polyglot books ---
    logger.info("Loading Polyglot opening books...")
    from sources.polyglot_books import load_book, get_weighted_moves

    import chess
    book_moves: dict[str, list] = {}
    polyglot_dir = Path(POLYGLOT_DIR)
    if polyglot_dir.exists():
        for bin_file in polyglot_dir.glob("*.bin"):
            try:
                with load_book(str(bin_file)) as reader:
                    # Get moves for opening positions
                    for opening in openings:
                        board = chess.Board(opening["fen"])
                        moves = get_weighted_moves(reader, board)
                        if moves:
                            book_moves[opening["fen"]] = moves
            except Exception as exc:
                logger.warning("Polyglot book %s failed: %s", bin_file, exc)

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

    chess960_positions = sample_chess960_positions(
        n=volume_override or 50_000, n_random_moves=20, rng=rng
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
    return config


def run_eval_splits(config: dict) -> frozenset[str]:
    """Generate eval splits and return the blocklist."""
    blocklist_path = EVAL_SPLITS_DIR / "blocklist.txt"

    if blocklist_path.exists():
        logger.info("Loading existing blocklist from %s", blocklist_path)
        return load_blocklist(str(blocklist_path))

    logger.info("Generating eval splits...")
    sources = {
        "perception": config.get("fen_pool", []),
        "rules": config.get("fen_pool", []),
        "tactics": config.get("puzzles", []),
        "evaluation": config.get("position_evals", []),
        "openings": config.get("openings", []),
        "endgames": config.get("endgame_positions", []),
        "planning": config.get("puzzles", []) + config.get("best_move_evals", []),
        "chess960": [e for e in config.get("fen_pool", []) if e.get("is_chess960")],
        "mate": config.get("mate_rows", []),
    }

    splits = generate_all_eval_splits(sources)
    save_eval_splits(splits, str(EVAL_SPLITS_DIR))
    return build_blocklist(splits)


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
        output_path = tier_dir / f"{task_id}.jsonl"

        # Resumption: skip if output exists and is non-empty
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info("Skipping %s — output exists at %s", task_id, output_path)
            continue

        logger.info("Generating %s (target: %d)...", task_id, gen.target_volume())

        with JSONLWriter(str(output_path)) as writer:
            for example in gen.generate():
                written = writer.write(example)
                stats.record(
                    task_id,
                    is_chess960=example.get("is_chess960", False),
                    passed_validation=written,
                )

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
    for d in (OUTPUT_DIR, POOL_DIR, EVAL_SPLITS_DIR, ANNOTATIONS_DIR, TIER_OUTPUT_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)

    stats = PipelineStats()

    if args.validate_only:
        blocklist_path = EVAL_SPLITS_DIR / "blocklist.txt"
        if not blocklist_path.exists():
            logger.error("No blocklist found at %s — run eval splits first", blocklist_path)
            return 1
        blocklist = load_blocklist(str(blocklist_path))
        report = audit_output_files(str(TIER_OUTPUT_DIR), blocklist)
        if report:
            logger.error("Contamination found:")
            for f, fens in report.items():
                logger.error("  %s: %d contaminated FENs", f, len(fens))
            return 1
        logger.info("No contamination found. All outputs clean.")
        return 0

    # Load sources
    config = load_sources(volume_override=args.volume)

    # Always run eval splits first
    blocklist = run_eval_splits(config)
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
