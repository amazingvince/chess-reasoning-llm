from chess_llm.sft.source_preparation import build_eval_split_sources


STANDARD_FEN = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"
CHESS960_FEN = "bqnnrkrb/pppppppp/8/8/8/8/PPPPPPPP/BQNNRKRB w KQkq - 0 1"


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
