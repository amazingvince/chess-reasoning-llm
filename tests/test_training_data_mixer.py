import json
from pathlib import Path


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _row(tier: int, fen: str, label: str, task: str | None = None):
    return {
        "task": task or f"{tier}.x_test",
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


def _pawn_fen(i: int, turn: str = "w") -> str:
    """Distinct valid FEN per (file index, turn): lone white pawn on rank 2."""
    left = str(i) if i else ""
    right = str(7 - i) if i != 7 else ""
    return f"4k3/8/8/8/8/8/{left}P{right}/4K3 {turn} - - 0 1"


def _phase(*tier_mix):
    from chess_llm.training.phases import PhaseConfig

    return PhaseConfig(
        name="test",
        display_name="Test",
        tier_mix=tier_mix,
        epochs=1,
        learning_rate=1e-4,
        warmup_ratio=0.0,
        weight_decay=0.0,
        resume_from=None,
    )


def _prompt_fen(row: dict) -> str:
    return row["messages"][1]["content"].split(": ", 1)[1]


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


def test_eval_fen_keys_persist_across_phase_builds(tmp_path):
    from chess_llm.training.data.mixer import build_phase_dataset
    from chess_llm.training.phases import TierMix

    for i in range(8):
        _append_jsonl(tmp_path / "tier1" / "rows.jsonl", _row(1, _pawn_fen(i), f"t1-{i}"))
    for i in range(8):
        _append_jsonl(
            tmp_path / "tier2" / "rows.jsonl", _row(2, _pawn_fen(i, "b"), f"t2-{i}")
        )

    _, first_eval = build_phase_dataset(
        _phase(TierMix(tier=1)),
        tmp_path,
        eval_fraction=0.25,
        seed=7,
    )
    first_eval_fens = {_prompt_fen(row) for row in first_eval}
    assert first_eval_fens

    state_path = tmp_path / "eval_fen_keys.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["tiers_seen"] == [1]

    second_train, second_eval = build_phase_dataset(
        _phase(TierMix(tier=1), TierMix(tier=2)),
        tmp_path,
        eval_fraction=0.25,
        seed=7,
    )
    second_eval_fens = {_prompt_fen(row) for row in second_eval}
    second_train_fens = {_prompt_fen(row) for row in second_train}

    assert first_eval_fens <= second_eval_fens
    assert not (first_eval_fens & second_train_fens)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["tiers_seen"] == [1, 2]


def test_eval_split_covers_every_task_with_data(tmp_path):
    from chess_llm.training.data.mixer import build_phase_dataset
    from chess_llm.training.phases import TierMix

    for i in range(8):
        _append_jsonl(
            tmp_path / "tier1" / "rows.jsonl",
            _row(1, _pawn_fen(i), f"a-{i}", task="1.1_fen_to_board"),
        )
    for i in range(4):
        _append_jsonl(
            tmp_path / "tier1" / "rows.jsonl",
            _row(1, _pawn_fen(i, "b"), f"a-{8 + i}", task="1.1_fen_to_board"),
        )
    for i in range(4, 8):
        _append_jsonl(
            tmp_path / "tier1" / "rows.jsonl",
            _row(1, _pawn_fen(i, "b"), f"b-{i}", task="1.2_board_to_fen"),
        )

    _, eval_ds = build_phase_dataset(
        _phase(TierMix(tier=1)),
        tmp_path,
        eval_fraction=0.1,
        seed=3,
    )

    assert set(eval_ds["task"]) == {"1.1_fen_to_board", "1.2_board_to_fen"}
    assert len(eval_ds) == 2


def test_review_tier_fraction_samples_eval_rows(tmp_path):
    from chess_llm.core.board import variant_fen_key
    from chess_llm.training.data.mixer import build_phase_dataset
    from chess_llm.training.phases import TierMix

    eval_fens = [_pawn_fen(i) for i in range(4)]
    train_fens = [_pawn_fen(i, "b") for i in range(4)]
    for i, fen in enumerate(eval_fens + train_fens):
        _append_jsonl(tmp_path / "tier1" / "rows.jsonl", _row(1, fen, f"r-{i}"))

    state = {
        "version": 1,
        "seed": 5,
        "eval_fraction": 0.5,
        "tiers_seen": [1],
        "eval_keys": sorted(variant_fen_key(fen) for fen in eval_fens),
    }
    (tmp_path / "eval_fen_keys.json").write_text(json.dumps(state), encoding="utf-8")

    train_ds, eval_ds = build_phase_dataset(
        _phase(TierMix(tier=1, fraction=0.5)),
        tmp_path,
        eval_fraction=0.5,
        seed=5,
    )

    assert len(eval_ds) == 2
    assert len(train_ds) == 2
    assert {_prompt_fen(row) for row in eval_ds} <= set(eval_fens)
    assert {_prompt_fen(row) for row in train_ds} <= set(train_fens)


def test_small_tier_fraction_selects_at_least_one_row(tmp_path):
    from chess_llm.training.data.mixer import build_phase_dataset, summarize_phase_data
    from chess_llm.training.phases import TierMix

    _append_jsonl(tmp_path / "tier1" / "rows.jsonl", _row(1, _pawn_fen(0), "a"))
    _append_jsonl(tmp_path / "tier1" / "rows.jsonl", _row(1, _pawn_fen(1), "b"))

    phase = _phase(TierMix(tier=1, fraction=0.1))
    train_ds, eval_ds = build_phase_dataset(phase, tmp_path, eval_fraction=0.0, seed=5)
    summary = summarize_phase_data(phase, tmp_path)

    assert len(train_ds) == 1
    assert len(eval_ds) == 0
    assert summary["tier_1"] == 1


def test_build_phase_dataset_computes_each_unique_fen_key_once(tmp_path, monkeypatch):
    from chess_llm.training.data import mixer
    from chess_llm.training.phases import TierMix

    fen_a = _pawn_fen(0)
    fen_b = _pawn_fen(1)
    for label, fen in (("a1", fen_a), ("a2", fen_a), ("a3", fen_a), ("b1", fen_b)):
        _append_jsonl(tmp_path / "tier1" / "rows.jsonl", _row(1, fen, label))

    calls: list[str] = []
    real_variant_fen_key = mixer.variant_fen_key

    def counting_variant_fen_key(fen, chess960=False):
        calls.append(fen)
        return real_variant_fen_key(fen, chess960=chess960)

    monkeypatch.setattr(mixer, "variant_fen_key", counting_variant_fen_key)

    mixer.build_phase_dataset(
        _phase(TierMix(tier=1)),
        tmp_path,
        eval_fraction=0.25,
        seed=5,
    )

    assert sorted(calls) == sorted([fen_a, fen_b])


def test_training_fen_key_preserves_chess_variant():
    from chess_llm.core.board import variant_fen_key
    from chess_llm.training.data.mixer import _fen_key

    assert _fen_key({"fen": STARTING_FEN, "is_chess960": False}) == variant_fen_key(
        STARTING_FEN
    )
    assert _fen_key(
        {"fen": STARTING_FEN, "metadata": {"chess960_id": 518}}
    ) == variant_fen_key(STARTING_FEN, chess960=True)


def _grid_fen(i: int, turn: str = "w") -> str:
    """Distinct valid FEN per (i, turn): lone white pawn on ranks 2..7."""
    rank = 2 + (i // 8) % 6
    file_index = i % 8
    left = str(file_index) if file_index else ""
    right = str(7 - file_index) if file_index != 7 else ""
    placement_rows = ["4k3", "8", "8", "8", "8", "8", "8", "4K3"]
    placement_rows[8 - rank] = f"{left}P{right}"
    return "/".join(placement_rows) + f" {turn} - - 0 1"


def _schedule(*segments):
    from chess_llm.training.schedule import ScheduleConfig

    return ScheduleConfig(
        name="schedule-test",
        display_name="Test schedule",
        segments=tuple(segments),
        learning_rate=1e-4,
        warmup_ratio=0.0,
        weight_decay=0.0,
    )


def _segment(name: str, fraction: float, weights):
    from chess_llm.training.schedule import ScheduleSegment

    return ScheduleSegment(
        name=name,
        fraction_of_run=fraction,
        tier_weights=tuple(weights),
    )


def _write_tier_rows(tmp_path, tier: int, count: int, turn: str = "w") -> None:
    for i in range(count):
        _append_jsonl(
            tmp_path / f"tier{tier}" / "rows.jsonl",
            _row(tier, _grid_fen(i, turn), f"t{tier}-{i}"),
        )


def test_build_schedule_dataset_orders_rows_by_segment_and_respects_total(tmp_path):
    from chess_llm.training.data.mixer import build_schedule_dataset

    _write_tier_rows(tmp_path, 1, 16)
    _write_tier_rows(tmp_path, 2, 16, "b")

    schedule = _schedule(
        _segment("first", 0.5, [(1, 1.0)]),
        _segment("second", 0.5, [(2, 1.0)]),
    )
    train_ds, eval_ds, plans = build_schedule_dataset(
        schedule,
        tmp_path,
        eval_fraction=0.0,
        seed=3,
        total_examples=20,
    )

    assert len(train_ds) == 20
    assert len(eval_ds) == 0
    assert [plan.name for plan in plans] == ["first", "second"]
    assert plans[-1].end_row == 20

    tiers = train_ds["tier"]
    for plan in plans:
        segment_tiers = set(tiers[plan.start_row:plan.end_row])
        assert segment_tiers == {tier for tier, _ in plan.tier_rows}

    assert "fen" not in train_ds.column_names
    assert "task" in train_ds.column_names
    assert "tier" in train_ds.column_names


def test_build_schedule_dataset_interleaves_tiers_within_segment(tmp_path):
    from chess_llm.training.data.mixer import build_schedule_dataset

    _write_tier_rows(tmp_path, 1, 16)
    _write_tier_rows(tmp_path, 2, 16, "b")

    schedule = _schedule(_segment("mix", 1.0, [(1, 0.5), (2, 0.5)]))
    train_ds, _, plans = build_schedule_dataset(
        schedule,
        tmp_path,
        eval_fraction=0.0,
        seed=3,
        total_examples=16,
    )

    tiers = train_ds["tier"]
    assert sorted(tiers) == [1] * 8 + [2] * 8
    # Rows are shuffled within the segment, not appended as tier blocks.
    assert tiers != sorted(tiers)
    transitions = sum(1 for a, b in zip(tiers, tiers[1:]) if a != b)
    assert transitions > 1
    assert plans[0].tier_rows == ((1, 8), (2, 8))


def test_build_schedule_dataset_is_deterministic(tmp_path):
    from chess_llm.training.data.mixer import build_schedule_dataset

    _write_tier_rows(tmp_path, 1, 12)
    _write_tier_rows(tmp_path, 2, 12, "b")

    schedule = _schedule(
        _segment("first", 0.5, [(1, 0.75), (2, 0.25)]),
        _segment("second", 0.5, [(1, 0.25), (2, 0.75)]),
    )

    first_train, _, first_plans = build_schedule_dataset(
        schedule, tmp_path, eval_fraction=0.25, seed=11,
    )
    second_train, _, second_plans = build_schedule_dataset(
        schedule, tmp_path, eval_fraction=0.25, seed=11,
    )

    assert first_plans == second_plans
    assert [_prompt_fen(row) for row in first_train] == [
        _prompt_fen(row) for row in second_train
    ]


def test_schedule_eval_covers_last_segment_tier_from_the_start(tmp_path):
    from chess_llm.training.data.mixer import build_schedule_dataset

    _write_tier_rows(tmp_path, 1, 16)
    _write_tier_rows(tmp_path, 7, 16, "b")

    # Tier 7 only enters the mix in the last segment, but its eval rows and
    # persisted eval FEN keys must exist from step 0.
    schedule = _schedule(
        _segment("early", 0.75, [(1, 1.0)]),
        _segment("late", 0.25, [(1, 0.5), (7, 0.5)]),
    )
    _, eval_ds, _ = build_schedule_dataset(
        schedule,
        tmp_path,
        eval_fraction=0.25,
        seed=7,
    )

    assert {1, 7} <= set(eval_ds["tier"])
    state = json.loads((tmp_path / "eval_fen_keys.json").read_text(encoding="utf-8"))
    assert state["tiers_seen"] == [1, 7]


def test_build_schedule_dataset_prevents_cross_tier_fen_leak(tmp_path):
    from chess_llm.training.data.mixer import build_schedule_dataset

    shared_fen = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"
    _append_jsonl(tmp_path / "tier1" / "rows.jsonl", _row(1, shared_fen, "t1"))
    _append_jsonl(tmp_path / "tier1" / "rows.jsonl", _row(1, shared_fen, "t1b"))
    _write_tier_rows(tmp_path, 1, 8)
    _append_jsonl(tmp_path / "tier2" / "rows.jsonl", _row(2, shared_fen, "t2"))
    _write_tier_rows(tmp_path, 2, 8, "b")

    schedule = _schedule(
        _segment("first", 0.5, [(1, 1.0)]),
        _segment("second", 0.5, [(1, 0.25), (2, 0.75)]),
    )
    train_ds, eval_ds, _ = build_schedule_dataset(
        schedule,
        tmp_path,
        eval_fraction=0.25,
        seed=5,
    )

    train_prompts = "\n".join(row["messages"][1]["content"] for row in train_ds)
    eval_prompts = "\n".join(row["messages"][1]["content"] for row in eval_ds)

    assert "fen" not in train_ds.column_names
    assert "fen" not in eval_ds.column_names
    assert not (shared_fen in train_prompts and shared_fen in eval_prompts)


def test_schedule_cycler_exhausts_pool_before_duplicating(tmp_path):
    from collections import Counter

    from chess_llm.training.data.mixer import build_schedule_dataset

    _write_tier_rows(tmp_path, 1, 6)

    # Demand of 8 rows against a pool of 6, drawn sequentially across two
    # segments: every pool row must appear before any row repeats.
    schedule = _schedule(
        _segment("first", 0.5, [(1, 1.0)]),
        _segment("second", 0.5, [(1, 1.0)]),
    )
    train_ds, _, _ = build_schedule_dataset(
        schedule,
        tmp_path,
        eval_fraction=0.0,
        seed=9,
        total_examples=8,
    )

    counts = Counter(_prompt_fen(row) for row in train_ds)
    assert len(train_ds) == 8
    assert len(counts) == 6
    assert max(counts.values()) == 2
    assert min(counts.values()) == 1


def test_build_schedule_dataset_defaults_total_to_pool_sizes(tmp_path):
    from chess_llm.training.data.mixer import build_schedule_dataset

    _write_tier_rows(tmp_path, 1, 10)
    _write_tier_rows(tmp_path, 2, 6, "b")

    schedule = _schedule(_segment("all", 1.0, [(1, 0.5), (2, 0.5)]))
    train_ds, _, plans = build_schedule_dataset(
        schedule,
        tmp_path,
        eval_fraction=0.0,
        seed=3,
    )

    assert len(train_ds) == 16
    assert plans[-1].end_row == 16


def test_summarize_schedule_data_reports_segments_and_boundaries(tmp_path):
    from chess_llm.training.data.mixer import summarize_schedule_data

    _write_tier_rows(tmp_path, 1, 16)
    _write_tier_rows(tmp_path, 2, 16, "b")

    schedule = _schedule(
        _segment("first", 0.5, [(1, 1.0)]),
        _segment("second", 0.5, [(1, 0.5), (2, 0.5)]),
    )
    summary = summarize_schedule_data(schedule, tmp_path, total_examples=64)

    assert summary["total_examples"] == 64
    assert summary["tier_pool_sizes"] == {"tier_1": 16, "tier_2": 16}
    assert summary["boundary_steps"] == [1, 2]
    assert summary["segments"][0]["tier_rows"] == {"tier_1": 32}
    assert summary["segments"][1]["tier_rows"] == {"tier_1": 16, "tier_2": 16}
    assert summary["segments"][1]["start_row"] == 32
    assert summary["segments"][1]["end_row"] == 64
