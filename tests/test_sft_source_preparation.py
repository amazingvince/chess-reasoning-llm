from chess_llm.sft.source_preparation import (
    EVAL_SPLIT_EXCLUDED_SOURCES,
    build_eval_split_sources,
)


STANDARD_FEN = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"
CHESS960_FEN = "bqnnrkrb/pppppppp/8/8/8/8/PPPPPPPP/BQNNRKRB w KQkq - 0 1"
SELF_PLAY_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def test_build_eval_split_sources_keeps_chess960_out_of_standard_splits():
    config = {
        "fen_pool": [
            {"fen": STANDARD_FEN, "source": "standard"},
            {"fen": CHESS960_FEN, "metadata": {"chess960_id": 3}},
        ],
        "openings": [],
        "position_evals": [],
    }

    prepared = build_eval_split_sources(config)

    assert prepared.sources["perception"] == [{"fen": STANDARD_FEN, "source": "standard"}]
    assert prepared.sources["rules"] == [{"fen": STANDARD_FEN, "source": "standard"}]
    assert prepared.sources["chess960"] == [
        {"fen": CHESS960_FEN, "metadata": {"chess960_id": 3}}
    ]


def test_build_eval_split_sources_partitions_openings_and_enriches_eval_copy():
    opening = {"fen": STANDARD_FEN, "eco": "A00", "name": "Unit Opening"}
    config = {
        "fen_pool": [],
        "openings": [opening, {"fen": "missing_eco"}],
        "book_moves": {STANDARD_FEN: [("e2e4", 10)]},
        "position_evals": [],
    }

    prepared = build_eval_split_sources(
        config,
        opening_holdout_fraction=1.0,
    )

    assert config["openings"] == [opening, {"fen": "missing_eco"}]
    assert prepared.train_openings == [{"fen": "missing_eco"}]
    assert prepared.eval_openings == [
        {
            "fen": STANDARD_FEN,
            "eco": "A00",
            "name": "Unit Opening",
            "book_moves": [("e2e4", 10)],
        }
    ]
    assert "book_moves" not in opening
    assert prepared.sources["openings"] == prepared.eval_openings


def test_build_eval_split_sources_excludes_self_play_rows_from_fen_pools():
    assert "self_play" in EVAL_SPLIT_EXCLUDED_SOURCES
    config = {
        "fen_pool": [
            {"fen": STANDARD_FEN, "source": "lichess_games"},
            {"fen": SELF_PLAY_FEN, "source": "self_play", "game_id": "abc"},
            {"fen": CHESS960_FEN, "source": "self_play", "metadata": {"chess960_id": 3}},
        ],
        "openings": [],
        "position_evals": [],
    }

    prepared = build_eval_split_sources(config)

    assert prepared.standard_fen_pool == [
        {"fen": STANDARD_FEN, "source": "lichess_games"}
    ]
    assert prepared.sources["perception"] == prepared.standard_fen_pool
    assert prepared.sources["rules"] == prepared.standard_fen_pool
    assert prepared.chess960_fen_pool == []
    assert prepared.sources["chess960"] == []


def test_eval_split_manifest_is_stable_when_self_play_rows_appear():
    from chess_llm.sft import pipeline

    base_pool = [{"fen": STANDARD_FEN, "source": "lichess_games"}]
    with_self_play = base_pool + [
        {"fen": SELF_PLAY_FEN, "source": "self_play", "game_id": "abc"}
    ]

    manifests = []
    for fen_pool in (base_pool, with_self_play):
        prepared = build_eval_split_sources(
            {"fen_pool": list(fen_pool), "openings": [], "position_evals": []}
        )
        manifests.append(
            pipeline.build_eval_split_manifest(
                prepared,
                split_sizes={"perception": 1, "rules": 1},
                volume_override=None,
            )
        )

    assert manifests[0] == manifests[1]


def test_build_eval_split_sources_filters_evaluation_by_depth():
    shallow = {"fen": "shallow", "depth": 39}
    deep = {"fen": "deep", "depth": 40}
    config = {
        "fen_pool": [],
        "openings": [],
        "position_evals": [shallow, deep],
    }

    prepared = build_eval_split_sources(config, min_depth_eval_benchmark=40)

    assert prepared.eval_benchmark_evals == [deep]
    assert prepared.sources["evaluation"] == [deep]
