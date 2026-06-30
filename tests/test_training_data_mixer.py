import json
from pathlib import Path


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _row(tier: int, fen: str, label: str):
    return {
        "task": f"{tier}.x_test",
        "tier": tier,
        "fen": fen,
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": f"{label}: {fen}"},
            {"role": "assistant", "content": "a"},
        ],
    }


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def test_package_build_phase_dataset_prevents_cross_tier_fen_leak(tmp_path):
    from chess_llm.training.data.mixer import build_phase_dataset
    from chess_llm.training.phases import PhaseConfig, TierMix

    shared_fen = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"
    _append_jsonl(tmp_path / "tier1" / "rows.jsonl", _row(1, shared_fen, "t1"))
    _append_jsonl(tmp_path / "tier1" / "rows.jsonl", _row(1, shared_fen, "t1b"))
    _append_jsonl(tmp_path / "tier2" / "rows.jsonl", _row(2, shared_fen, "t2"))
    _append_jsonl(
        tmp_path / "tier2" / "rows.jsonl",
        _row(2, "8/8/8/8/8/8/4K3/4k3 b - - 0 1", "other"),
    )

    phase = PhaseConfig(
        name="test",
        display_name="Test",
        tier_mix=(TierMix(tier=1), TierMix(tier=2)),
        epochs=1,
        learning_rate=1e-4,
        warmup_ratio=0.0,
        weight_decay=0.0,
        resume_from=None,
    )

    train_ds, eval_ds = build_phase_dataset(
        phase,
        tmp_path,
        eval_fraction=0.25,
        seed=5,
    )

    train_prompts = "\n".join(row["messages"][1]["content"] for row in train_ds)
    eval_prompts = "\n".join(row["messages"][1]["content"] for row in eval_ds)

    assert "fen" not in train_ds.column_names
    assert "fen" not in eval_ds.column_names
    assert not (shared_fen in train_prompts and shared_fen in eval_prompts)


def test_package_build_phase_dataset_upsamples_selected_task_train_only(tmp_path):
    from chess_llm.training.data.mixer import build_phase_dataset, summarize_phase_data
    from chess_llm.training.phases import PhaseConfig, TierMix

    phase = PhaseConfig(
        name="test",
        display_name="Test",
        tier_mix=(TierMix(tier=1),),
        epochs=1,
        learning_rate=1e-4,
        warmup_ratio=0.0,
        weight_decay=0.0,
        resume_from=None,
    )
    state_row = _row(1, "8/8/8/8/8/8/4K3/4k3 w - - 0 1", "state")
    state_row["task"] = "1.5_state_tracking"
    other_row = _row(1, "8/8/8/8/8/8/4K3/4k3 b - - 0 1", "other")
    other_row["task"] = "1.1_fen_to_board"
    _append_jsonl(tmp_path / "tier1" / "rows.jsonl", state_row)
    _append_jsonl(tmp_path / "tier1" / "rows.jsonl", other_row)

    train_ds, eval_ds = build_phase_dataset(
        phase,
        tmp_path,
        eval_fraction=0.0,
        task_upsample={"1.5_state_tracking": 4},
        seed=5,
    )
    summary = summarize_phase_data(
        phase,
        tmp_path,
        task_upsample={"1.5_state_tracking": 4},
    )

    assert len(eval_ds) == 0
    assert train_ds["task"].count("1.5_state_tracking") == 4
    assert train_ds["task"].count("1.1_fen_to_board") == 1
    assert summary["tier_1"] == 5
    assert summary["total"] == 5


def test_training_fen_key_preserves_chess_variant():
    from chess_llm.core.board import variant_fen_key
    from chess_llm.training.data.mixer import _fen_key

    assert _fen_key({"fen": STARTING_FEN, "is_chess960": False}) == variant_fen_key(
        STARTING_FEN
    )
    assert _fen_key(
        {"fen": STARTING_FEN, "metadata": {"chess960_id": 518}}
    ) == variant_fen_key(STARTING_FEN, chess960=True)
