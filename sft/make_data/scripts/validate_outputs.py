#!/usr/bin/env python3
"""Post-hoc validation of generated JSONL files.

Usage:
    python validate_outputs.py
    python validate_outputs.py --output-dir E:/chess_sft_data/output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import TIER_OUTPUT_DIR, EVAL_SPLITS_DIR
from validation.validator import validate_example
from validation.decontamination import audit_output_files
from pool.eval_split import load_blocklist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated JSONL outputs")
    parser.add_argument(
        "--output-dir", type=str, default=str(TIER_OUTPUT_DIR),
        help="Directory containing tier output JSONL files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logger.error("Output directory does not exist: %s", output_dir)
        return 1

    # 1. Validate each JSONL file
    total_examples = 0
    total_errors = 0
    file_stats: dict[str, dict] = {}

    for jsonl_path in sorted(output_dir.rglob("*.jsonl")):
        file_count = 0
        file_errors = 0
        errors_sample: list[str] = []

        with open(jsonl_path, encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                except json.JSONDecodeError as e:
                    file_errors += 1
                    if len(errors_sample) < 5:
                        errors_sample.append(f"Line {line_num}: JSON parse error: {e}")
                    continue

                file_count += 1
                passed, errs = validate_example(example)
                if not passed:
                    file_errors += 1
                    if len(errors_sample) < 5:
                        errors_sample.append(
                            f"Line {line_num}: {'; '.join(errs)}"
                        )

        total_examples += file_count
        total_errors += file_errors
        rel_path = str(jsonl_path.relative_to(output_dir))
        file_stats[rel_path] = {
            "count": file_count,
            "errors": file_errors,
            "error_rate": f"{file_errors/file_count*100:.2f}%" if file_count else "n/a",
        }

        if errors_sample:
            logger.warning("%s: %d errors (sample below)", rel_path, file_errors)
            for e in errors_sample:
                logger.warning("  %s", e)
        else:
            logger.info("%s: %d examples, 0 errors", rel_path, file_count)

    # 2. Decontamination check
    blocklist_path = EVAL_SPLITS_DIR / "blocklist.txt"
    if blocklist_path.exists():
        logger.info("Running decontamination audit...")
        blocklist = load_blocklist(str(blocklist_path))
        contamination = audit_output_files(args.output_dir, blocklist)
        if contamination:
            logger.error("CONTAMINATION FOUND:")
            for f, fens in contamination.items():
                logger.error("  %s: %d contaminated FENs", f, len(fens))
                total_errors += len(fens)
        else:
            logger.info("Decontamination check passed: 0 contaminated FENs")
    else:
        logger.warning("No blocklist found — skipping decontamination check")

    # Summary
    print(f"\n{'='*60}")
    print(f"Validation Summary")
    print(f"{'='*60}")
    print(f"Total files:    {len(file_stats)}")
    print(f"Total examples: {total_examples:,}")
    print(f"Total errors:   {total_errors:,}")
    print(f"Error rate:     {total_errors/total_examples*100:.3f}%" if total_examples else "n/a")
    print(f"{'='*60}")

    for rel_path, info in sorted(file_stats.items()):
        status = "PASS" if info["errors"] == 0 else "FAIL"
        print(f"  [{status}] {rel_path}: {info['count']:,} examples, {info['errors']} errors ({info['error_rate']})")

    return 1 if total_errors > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
