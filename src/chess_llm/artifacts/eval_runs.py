"""Utilities for durable evaluation run ledgers and artifact mirrors."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def finalize_evaluation_artifacts(
    eval_run_path: str | Path,
    *,
    ledger_path: str | Path | None = None,
    mirror_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Update one eval-run sidecar, mirror files, and append a ledger row."""
    path = Path(eval_run_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        payload["metadata"] = metadata

    if mirror_dir is not None:
        run_id = str(payload["run_id"])
        destination_dir = Path(mirror_dir) / run_id
        mirrored = _planned_mirrored_artifacts(payload, path, destination_dir)
        metadata["artifact_mirror_dir"] = str(destination_dir)
        metadata["mirrored_artifacts"] = mirrored

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if mirror_dir is not None:
        _mirror_evaluation_files(payload, path, Path(mirror_dir) / str(payload["run_id"]))

    if ledger_path is not None:
        _append_ledger_row(Path(ledger_path), payload)

    return payload


def _planned_mirrored_artifacts(
    payload: dict[str, Any],
    eval_run_path: Path,
    destination_dir: Path,
) -> dict[str, str]:
    return {
        label: str(destination_dir / source.name)
        for label, source in _evaluation_artifact_sources(payload, eval_run_path).items()
        if source is not None and source.exists()
    }


def _mirror_evaluation_files(
    payload: dict[str, Any],
    eval_run_path: Path,
    destination_dir: Path,
) -> dict[str, str]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    mirrored: dict[str, str] = {}

    for label, source in _evaluation_artifact_sources(payload, eval_run_path).items():
        if source is None or not source.exists():
            continue
        destination = destination_dir / source.name
        shutil.copy2(source, destination)
        mirrored[label] = str(destination)
    return mirrored


def _evaluation_artifact_sources(
    payload: dict[str, Any],
    eval_run_path: Path,
) -> dict[str, Path | None]:
    predictions = _path_or_none(payload.get("predictions_path"))
    results = _path_or_none(payload.get("results_path"))
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    analysis = _path_or_none(metadata.get("prediction_analysis_path"))

    multipv_cache = None
    if predictions is not None:
        multipv_cache = predictions.with_suffix(".multipv.sqlite")

    return {
        "predictions": predictions,
        "results": results,
        "analysis": analysis,
        "eval_run": eval_run_path,
        "multipv_cache": multipv_cache,
    }


def _append_ledger_row(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _path_or_none(value: object) -> Path | None:
    if value is None:
        return None
    return Path(str(value))
