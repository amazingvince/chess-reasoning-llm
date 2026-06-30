from __future__ import annotations

from pathlib import Path


def test_packaged_source_does_not_embed_local_machine_paths():
    package_root = Path(__file__).resolve().parents[1] / "src" / "chess_llm"
    forbidden = (
        "E:/",
        "C:/Users/",
        "/home/amazi",
        "/mnt/e",
        "Downloads/stockfish",
    )
    offenders: list[str] = []

    for path in sorted(package_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden:
            if fragment in text.replace("\\", "/"):
                offenders.append(f"{path.relative_to(package_root)} contains {fragment!r}")

    assert offenders == []
