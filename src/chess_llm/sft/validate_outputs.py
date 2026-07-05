"""Post-hoc validation of generated SFT JSONL output files."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from chess_llm.sft.completeness import audit_output_completeness
from chess_llm.sft.decontamination import audit_output_files
from chess_llm.sft.eval_split import load_blocklist
from chess_llm.sft.files import iter_jsonl_artifacts
from chess_llm.sft.settings import SftDataSettings
from chess_llm.sft.validation import validate_example

logger = logging.getLogger(__name__)

_SETTINGS_ROOT = Path.cwd()
SETTINGS = SftDataSettings.from_env(_SETTINGS_ROOT)

TIER_OUTPUT_DIR = SETTINGS.tier_output_dir
EVAL_SPLITS_DIR = SETTINGS.eval_splits_dir


@dataclass(frozen=True)
class ValidationRunResult:
    """Structured result from a post-hoc output validation run."""

    exit_code: int
    total_files: int
    total_examples: int
    total_errors: int
    file_stats: dict[str, dict]
    completeness_issues: dict[str, str]
    contamination: dict[str, list[str]]


def validate_outputs(
    output_dir: str | Path = TIER_OUTPUT_DIR,
    *,
    expected_volume: int | None = None,
    skip_completeness: bool = False,
    skip_decontamination: bool = False,
    blocklist_path: str | Path | None = None,
    tiers: Sequence[int] | None = None,
    task_ids: Sequence[str] | None = None,
) -> ValidationRunResult:
    """Validate generated JSONL outputs and return structured counts."""
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.error("Output directory does not exist: %s", output_path)
        return ValidationRunResult(
            exit_code=1,
            total_files=0,
            total_examples=0,
            total_errors=1,
            file_stats={},
            completeness_issues={},
            contamination={},
        )

    total_examples = 0
    total_errors = 0
    file_stats: dict[str, dict] = {}

    for jsonl_path in iter_jsonl_artifacts(output_path):
        if not _selected_output_file(
            jsonl_path,
            output_path,
            tiers=tiers,
            task_ids=task_ids,
        ):
            continue
        file_count, file_errors, errors_sample = _validate_jsonl_file(jsonl_path)
        total_examples += file_count
        total_errors += file_errors

        rel_path = jsonl_path.relative_to(output_path).as_posix()
        file_stats[rel_path] = {
            "count": file_count,
            "errors": file_errors,
            "error_rate": _format_error_rate(file_errors, file_count),
        }

        if errors_sample:
            logger.warning("%s: %d errors (sample below)", rel_path, file_errors)
            for error in errors_sample:
                logger.warning("  %s", error)
        else:
            logger.info("%s: %d examples, 0 errors", rel_path, file_count)

    completeness_issues: dict[str, str] = {}
    if not skip_completeness:
        logger.info("Running completeness audit...")
        completeness_issues = _normalize_issue_paths(
            audit_output_completeness(
                output_path,
                expected_volume_override=expected_volume,
                tiers=tiers,
                task_ids=task_ids,
            )
        )
        if completeness_issues:
            logger.error("COMPLETENESS ISSUES FOUND:")
            for rel_path, issue in completeness_issues.items():
                logger.error("  %s: %s", rel_path, issue)
            total_errors += len(completeness_issues)
        else:
            logger.info("Completeness check passed: all expected task files present")

    resolved_blocklist = (
        Path(blocklist_path)
        if blocklist_path is not None
        else Path(EVAL_SPLITS_DIR) / "blocklist.txt"
    )
    contamination: dict[str, list[str]] = {}
    if skip_decontamination:
        logger.warning("Skipping decontamination check by explicit request")
    elif resolved_blocklist.exists():
        logger.info("Running decontamination audit...")
        blocklist = load_blocklist(resolved_blocklist)
        contamination = audit_output_files(output_path, blocklist)
        if contamination:
            logger.error("CONTAMINATION FOUND:")
            for file_path, fens in contamination.items():
                logger.error("  %s: %d contaminated FENs", file_path, len(fens))
                total_errors += len(fens)
        else:
            logger.info("Decontamination check passed: 0 contaminated FENs")
    else:
        logger.error(
            "No blocklist found at %s; pass --skip-decontamination only for "
            "intentional local smoke/debug validation",
            resolved_blocklist,
        )
        total_errors += 1

    return ValidationRunResult(
        exit_code=1 if total_errors > 0 else 0,
        total_files=len(file_stats),
        total_examples=total_examples,
        total_errors=total_errors,
        file_stats=file_stats,
        completeness_issues=completeness_issues,
        contamination=contamination,
    )


def _selected_output_file(
    jsonl_path: Path,
    output_path: Path,
    *,
    tiers: Sequence[int] | None,
    task_ids: Sequence[str] | None,
) -> bool:
    if tiers is not None:
        rel_parts = jsonl_path.relative_to(output_path).parts
        tier_part = rel_parts[0] if rel_parts else ""
        selected_tiers = {f"tier{tier}" for tier in tiers}
        if tier_part not in selected_tiers:
            return False
    if task_ids is not None and jsonl_path.stem not in set(task_ids):
        return False
    return True


def _validate_jsonl_file(jsonl_path: Path) -> tuple[int, int, list[str]]:
    file_count = 0
    file_errors = 0
    errors_sample: list[str] = []

    with jsonl_path.open(encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                file_errors += 1
                if len(errors_sample) < 5:
                    errors_sample.append(f"Line {line_num}: JSON parse error: {exc}")
                continue

            file_count += 1
            passed, errors = validate_example(example)
            if not passed:
                file_errors += 1
                if len(errors_sample) < 5:
                    errors_sample.append(f"Line {line_num}: {'; '.join(errors)}")

    return file_count, file_errors, errors_sample


def _format_error_rate(errors: int, count: int) -> str:
    return f"{errors / count * 100:.2f}%" if count else "n/a"


def _normalize_issue_paths(issues: dict[str, str]) -> dict[str, str]:
    return {Path(path).as_posix(): issue for path, issue in issues.items()}


def print_summary(result: ValidationRunResult) -> None:
    """Print the legacy validation summary format."""
    print(f"\n{'=' * 60}")
    print("Validation Summary")
    print(f"{'=' * 60}")
    print(f"Total files:    {result.total_files}")
    print(f"Total examples: {result.total_examples:,}")
    print(f"Total errors:   {result.total_errors:,}")
    if result.total_examples:
        print(f"Error rate:     {result.total_errors / result.total_examples * 100:.3f}%")
    else:
        print("Error rate:     n/a")
    print(f"{'=' * 60}")

    for rel_path, info in sorted(result.file_stats.items()):
        status = "PASS" if info["errors"] == 0 else "FAIL"
        print(
            f"  [{status}] {rel_path}: {info['count']:,} examples, "
            f"{info['errors']} errors ({info['error_rate']})"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate generated JSONL outputs")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(TIER_OUTPUT_DIR),
        help="Directory containing tier output JSONL files",
    )
    parser.add_argument(
        "--expected-volume",
        type=int,
        default=None,
        help="Expected rows per task, for smoke outputs generated with --volume",
    )
    parser.add_argument(
        "--skip-completeness",
        action="store_true",
        help="Skip checks for missing or underfilled task files",
    )
    parser.add_argument(
        "--skip-decontamination",
        action="store_true",
        help="Skip eval blocklist contamination checks for intentional local smoke/debug validation",
    )
    parser.add_argument(
        "--blocklist-path",
        "--blocklist",
        type=str,
        default=None,
        help="Eval blocklist path for decontamination checks",
    )
    parser.add_argument(
        "--tier",
        type=int,
        nargs="+",
        default=None,
        help="Restrict completeness and file validation to selected tier(s)",
    )
    parser.add_argument(
        "--task",
        dest="task_ids",
        type=str,
        nargs="+",
        default=None,
        help="Restrict completeness and file validation to selected task id(s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_arg_parser().parse_args(argv)
    result = validate_outputs(
        args.output_dir,
        expected_volume=args.expected_volume,
        skip_completeness=args.skip_completeness,
        skip_decontamination=args.skip_decontamination,
        blocklist_path=args.blocklist_path,
        tiers=args.tier,
        task_ids=args.task_ids,
    )
    print_summary(result)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVAL_SPLITS_DIR",
    "SETTINGS",
    "TIER_OUTPUT_DIR",
    "ValidationRunResult",
    "build_arg_parser",
    "main",
    "print_summary",
    "_selected_output_file",
    "validate_outputs",
]
