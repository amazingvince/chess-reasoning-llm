#!/usr/bin/env python3
"""Extract Polyglot .bin files from archives in the polyglot_opening_books dir.

Handles .zip and .7z archives. Extracts all .bin files to the same directory.

Usage:
    python scripts/extract_polyglot_books.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import POLYGLOT_DIR


def extract_zip(archive: Path, dest: Path) -> int:
    """Extract .bin files from a .zip archive. Returns count extracted."""
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".bin"):
                # Extract to dest, flattening directory structure
                out = dest / Path(name).name
                with zf.open(name) as src, open(out, "wb") as dst:
                    dst.write(src.read())
                print(f"  Extracted: {out.name}")
                count += 1
    return count


def extract_7z(archive: Path, dest: Path) -> int:
    """Extract .bin files from a .7z archive. Returns count extracted."""
    try:
        import py7zr
    except ImportError:
        print(f"  SKIP {archive.name}: install py7zr (`pip install py7zr`) for .7z support")
        return 0
    count = 0
    with py7zr.SevenZipFile(archive, "r") as sz:
        all_files = sz.getnames()
        bin_files = [n for n in all_files if n.lower().endswith(".bin")]
        if not bin_files:
            return 0
        sz.extract(dest, bin_files)
        for name in bin_files:
            extracted = dest / name
            # Flatten: if extracted into a subdirectory, move to dest root
            target = dest / Path(name).name
            if extracted != target:
                extracted.rename(target)
            print(f"  Extracted: {target.name}")
            count += 1
    return count


def main() -> int:
    polyglot_dir = Path(POLYGLOT_DIR)
    if not polyglot_dir.exists():
        print(f"Polyglot directory not found: {polyglot_dir}")
        return 1

    total = 0
    for archive in sorted(polyglot_dir.iterdir()):
        if archive.suffix.lower() == ".zip":
            print(f"Processing {archive.name}...")
            total += extract_zip(archive, polyglot_dir)
        elif archive.suffix.lower() == ".7z":
            print(f"Processing {archive.name}...")
            total += extract_7z(archive, polyglot_dir)

    existing_bins = list(polyglot_dir.glob("*.bin"))
    print(f"\nDone. Extracted {total} new file(s). Total .bin files: {len(existing_bins)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
