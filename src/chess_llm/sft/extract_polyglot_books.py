"""Extract Polyglot opening-book ``.bin`` files from local archives."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from chess_llm.sft.settings import SftDataSettings

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_MAKE_DATA_ROOT = _REPO_ROOT / "sft" / "make_data"
_SETTINGS_ROOT = _LEGACY_MAKE_DATA_ROOT if _LEGACY_MAKE_DATA_ROOT.exists() else Path.cwd()
SETTINGS = SftDataSettings.from_env(_SETTINGS_ROOT)
POLYGLOT_DIR = SETTINGS.polyglot_dir


@dataclass(frozen=True)
class PolyglotExtractionResult:
    """Structured result from a Polyglot archive extraction run."""

    exit_code: int
    extracted_count: int
    total_bin_files: int
    processed_archives: list[Path]


def extract_zip(archive: Path, dest: Path) -> int:
    """Extract ``.bin`` files from a ZIP archive into ``dest``."""
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            output_name = _safe_bin_basename(name)
            if output_name is None:
                continue
            output_path = dest / output_name
            with zf.open(name) as src, output_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"  Extracted: {output_path.name}")
            count += 1
    return count


def extract_7z(archive: Path, dest: Path, seven_zip_cls: Any | None = None) -> int:
    """Extract ``.bin`` files from a 7z archive into ``dest``."""
    if seven_zip_cls is None:
        try:
            import py7zr
        except ImportError:
            print(
                f"  SKIP {archive.name}: install py7zr (`pip install py7zr`) "
                "for .7z support"
            )
            return 0
        seven_zip_cls = py7zr.SevenZipFile

    count = 0
    with seven_zip_cls(archive, "r") as sz:
        bin_files = [
            name
            for name in sz.getnames()
            if _safe_bin_basename(name) is not None and _is_safe_extract_target(name)
        ]
        if not bin_files:
            return 0
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sz.extract(path=tmp_path, targets=bin_files)
            for name in bin_files:
                output_name = _safe_bin_basename(name)
                if output_name is None:
                    continue
                extracted = tmp_path / _normalized_member_path(name)
                if not extracted.exists():
                    continue
                target = dest / output_name
                shutil.copy2(extracted, target)
                print(f"  Extracted: {target.name}")
                count += 1
    return count


def extract_polyglot_books(
    polyglot_dir: str | Path = POLYGLOT_DIR,
) -> PolyglotExtractionResult:
    """Extract local Polyglot book archives under ``polyglot_dir``."""
    base = Path(polyglot_dir)
    if not base.exists():
        print(f"Polyglot directory not found: {base}")
        return PolyglotExtractionResult(
            exit_code=1,
            extracted_count=0,
            total_bin_files=0,
            processed_archives=[],
        )

    total = 0
    processed: list[Path] = []
    for archive in sorted(base.iterdir()):
        suffix = archive.suffix.lower()
        if suffix == ".zip":
            print(f"Processing {archive.name}...")
            processed.append(archive)
            total += extract_zip(archive, base)
        elif suffix == ".7z":
            print(f"Processing {archive.name}...")
            processed.append(archive)
            total += extract_7z(archive, base)

    existing_bins = list(base.glob("*.bin"))
    print(f"\nDone. Extracted {total} new file(s). Total .bin files: {len(existing_bins)}")
    return PolyglotExtractionResult(
        exit_code=0,
        extracted_count=total,
        total_bin_files=len(existing_bins),
        processed_archives=processed,
    )


def _safe_bin_basename(name: str) -> str | None:
    basename = _normalized_member_path(name).name
    if not basename or basename in (".", ".."):
        return None
    if not basename.lower().endswith(".bin"):
        return None
    return basename


def _normalized_member_path(name: str) -> Path:
    return Path(*[part for part in name.replace("\\", "/").split("/") if part])


def _is_safe_extract_target(name: str) -> bool:
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return False
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return False
    if parts[0].endswith(":"):
        return False
    return all(part not in (".", "..") for part in parts)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Polyglot .bin files from local archives"
    )
    parser.add_argument(
        "--polyglot-dir",
        type=Path,
        default=Path(POLYGLOT_DIR),
        help="Directory containing .zip/.7z Polyglot book archives",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return extract_polyglot_books(args.polyglot_dir).exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "POLYGLOT_DIR",
    "PolyglotExtractionResult",
    "build_arg_parser",
    "extract_7z",
    "extract_polyglot_books",
    "extract_zip",
    "main",
]
