"""Export local SFT checkpoints into vLLM-compatible layouts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from chess_llm.training.phases import DEFAULT_BASE_MODEL

QWEN35_TEXT_MODEL_TYPE = "qwen3_5_text"
QWEN35_TEXT_ARCH = "Qwen3_5ForCausalLM"
QWEN35_WRAPPER_EXPORT_DIRNAME = "vllm_qwen35_wrapper"

_CHECKPOINT_OVERLAY_FILES = (
    "chat_template.jinja",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads((path / "config.json").read_text(encoding="utf-8"))


def _is_qwen35_text_checkpoint(config: dict[str, Any]) -> bool:
    architectures = tuple(config.get("architectures") or ())
    return (
        config.get("model_type") == QWEN35_TEXT_MODEL_TYPE
        and QWEN35_TEXT_ARCH in architectures
    )


def _link_or_copy(source: Path, target: Path) -> None:
    source = source.resolve()
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
        return
    except OSError:
        pass
    try:
        target.symlink_to(source)
        return
    except OSError:
        pass
    shutil.copy2(source, target)


def _copy_base_files(base_dir: Path, output_dir: Path) -> None:
    for source in base_dir.iterdir():
        if source.name.endswith(".safetensors"):
            continue
        if source.name == "model.safetensors.index.json":
            continue
        if source.is_file() or source.is_symlink():
            _link_or_copy(source, output_dir / source.name)


def _overlay_checkpoint_files(checkpoint_dir: Path, output_dir: Path) -> None:
    for name in _CHECKPOINT_OVERLAY_FILES:
        source = checkpoint_dir / name
        if source.exists():
            _link_or_copy(source, output_dir / name)


def _single_safetensors_weight_map(path: Path, target_name: str) -> dict[str, str]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "safetensors is required to export Qwen3.5 checkpoints for vLLM."
        ) from exc

    with safe_open(path, framework="pt") as handle:
        return {key: target_name for key in handle.keys()}


def _checkpoint_weight_map(
    checkpoint_dir: Path,
    *,
    target_prefix: str,
) -> tuple[dict[str, str], dict[str, Path]]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        source_to_target = {
            source_name: f"{target_prefix}{source_name}"
            for source_name in sorted(set(index["weight_map"].values()))
        }
        weight_map = {
            key: source_to_target[source_name]
            for key, source_name in index["weight_map"].items()
        }
        files = {
            target_name: checkpoint_dir / source_name
            for source_name, target_name in source_to_target.items()
        }
        return weight_map, files

    model_path = checkpoint_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_dir} does not contain model.safetensors or "
            "model.safetensors.index.json"
        )
    target_name = f"{target_prefix}model.safetensors"
    return (
        _single_safetensors_weight_map(model_path, target_name),
        {target_name: model_path},
    )


def _source_manifest(files: dict[str, Path]) -> dict[str, dict[str, int | str]]:
    manifest: dict[str, dict[str, int | str]] = {}
    for name, path in sorted(files.items()):
        stat = path.stat()
        manifest[name] = {
            "source": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return manifest


def _export_is_current(output_dir: Path, manifest: dict[str, Any]) -> bool:
    manifest_path = output_dir / "vllm_export_manifest.json"
    index_path = output_dir / "model.safetensors.index.json"
    config_path = output_dir / "config.json"
    if not manifest_path.exists() or not index_path.exists() or not config_path.exists():
        return False
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return existing == manifest


def export_qwen35_text_checkpoint_for_vllm(
    checkpoint_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    base_model: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Export a text-only Qwen3.5 SFT checkpoint as a vLLM wrapper checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    if output_dir is None:
        output_dir = checkpoint_dir / QWEN35_WRAPPER_EXPORT_DIRNAME
    output_dir = Path(output_dir)
    if output_dir.resolve() == checkpoint_dir.resolve():
        raise ValueError("output_dir must be different from checkpoint_dir")

    config = _load_config(checkpoint_dir)
    if not _is_qwen35_text_checkpoint(config):
        raise ValueError(
            f"{checkpoint_dir} is not a text-only Qwen3.5 CausalLM checkpoint"
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "huggingface_hub is required to export Qwen3.5 checkpoints for vLLM."
        ) from exc

    resolved_base_model = (
        base_model or os.environ.get("CHESS_SFT_BASE_MODEL") or DEFAULT_BASE_MODEL
    )
    base_dir = Path(snapshot_download(resolved_base_model))
    base_index_path = base_dir / "model.safetensors.index.json"
    if not base_index_path.exists():
        raise FileNotFoundError(
            f"Base model {resolved_base_model} does not provide "
            "model.safetensors.index.json"
        )

    base_index = json.loads(base_index_path.read_text(encoding="utf-8"))
    base_weight_map = dict(base_index["weight_map"])
    sft_weight_map, sft_files = _checkpoint_weight_map(
        checkpoint_dir,
        target_prefix="sft_",
    )
    missing_keys = sorted(set(sft_weight_map) - set(base_weight_map))
    if missing_keys:
        preview = ", ".join(missing_keys[:5])
        raise ValueError(
            "SFT checkpoint contains keys missing from the base Qwen3.5 model: "
            f"{preview}"
        )

    base_source_to_target = {
        source_name: f"base_{source_name}"
        for source_name in sorted(set(base_weight_map.values()))
    }
    base_files = {
        target_name: base_dir / source_name
        for source_name, target_name in base_source_to_target.items()
    }
    export_weight_map = {
        key: sft_weight_map.get(key, base_source_to_target[source_name])
        for key, source_name in base_weight_map.items()
    }

    source_files = {**base_files, **sft_files}
    manifest = {
        "format": "qwen35_text_vllm_wrapper_v1",
        "base_model": resolved_base_model,
        "base_dir": str(base_dir.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "sources": _source_manifest(source_files),
    }
    if output_dir.exists() and not overwrite and _export_is_current(output_dir, manifest):
        return output_dir

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _copy_base_files(base_dir, output_dir)
    _overlay_checkpoint_files(checkpoint_dir, output_dir)
    for target_name, source_path in source_files.items():
        _link_or_copy(source_path, output_dir / target_name)

    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": base_index.get("metadata", {}),
                "weight_map": export_weight_map,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "vllm_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir


def prepare_model_for_vllm(
    model_path: str,
    *,
    base_model: str | None = None,
    output_dir: Path | str | None = None,
) -> str:
    """Return a vLLM-loadable model path, exporting local Qwen3.5 SFT if needed."""
    path = Path(model_path)
    if not path.exists() or not path.is_dir() or not (path / "config.json").exists():
        return model_path

    try:
        config = _load_config(path)
    except (OSError, json.JSONDecodeError):
        return model_path
    if not _is_qwen35_text_checkpoint(config):
        return model_path

    return str(
        export_qwen35_text_checkpoint_for_vllm(
            path,
            output_dir=output_dir,
            base_model=base_model,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Qwen3.5 text-only SFT checkpoint for vLLM eval"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "Base Qwen3.5 wrapper model used for config/visual weights "
            f"(default: CHESS_SFT_BASE_MODEL or {DEFAULT_BASE_MODEL})"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = export_qwen35_text_checkpoint_for_vllm(
        args.checkpoint,
        output_dir=args.output_dir,
        base_model=args.base_model,
        overwrite=args.overwrite,
    )
    print(output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
