from __future__ import annotations

from chess_llm.sft.source_readiness import build_source_readiness_report


STANDARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
CHESS960_FEN = "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w KQkq - 0 1"


def test_readiness_report_counts_sources_and_tier_required_gaps():
    report = build_source_readiness_report(
        {
            "fen_pool": [
                {"fen": STANDARD_FEN, "is_chess960": False},
                {
                    "fen": CHESS960_FEN,
                    "is_chess960": True,
                    "metadata": {"chess960_id": 7},
                },
            ],
            "position_evals": [],
        },
        tiers=[4],
        strict_eval_splits=False,
    )

    assert report.counts["fen_pool"] == 2
    assert report.counts["fen_pool_standard"] == 1
    assert report.counts["fen_pool_chess960"] == 1
    assert report.ok is False
    assert [issue.source_key for issue in report.missing_required] == [
        "position_evals"
    ]
    assert report.to_dict()["missing_required"][0]["required_by"] == ["tier4"]


def test_optional_tier_sources_do_not_make_report_fail():
    report = build_source_readiness_report(
        {
            "fen_pool": [{"fen": STANDARD_FEN, "is_chess960": False}],
            "puzzles": [{"fen": STANDARD_FEN, "moves": "e2e4 e7e5"}],
            "position_evals": [{"fen": STANDARD_FEN, "best_move": "e2e4", "depth": 22}],
            "best_move_evals": [{"fen": STANDARD_FEN, "best_move": "e2e4", "depth": 22}],
            "consequence_evals": [{"fen": STANDARD_FEN, "best_move": "e2e4", "depth": 22}],
            "game_positions": [
                {
                    "fen": STANDARD_FEN,
                    "move_history": "e2e4",
                    "move_played_uci": "e7e5",
                }
            ],
            "mate_rows": [],
        },
        tiers=[7],
        strict_eval_splits=False,
    )

    assert report.ok is True
    assert [issue.source_key for issue in report.optional_gaps] == [
        "candidate_rating_evals",
        "mate_rows",
    ]
    optional_by_source = {
        item["source_key"]: item["required_by"]
        for item in report.to_dict()["optional_gaps"]
    }
    assert optional_by_source == {
        "candidate_rating_evals": ["tier7_r4_optional"],
        "mate_rows": ["tier7_optional"],
    }


def test_self_play_count_is_informational_only():
    report = build_source_readiness_report(
        {
            "fen_pool": [
                {"fen": STANDARD_FEN, "source": "lichess_games"},
                {"fen": STANDARD_FEN, "source": "self_play", "game_id": "abc"},
            ],
        },
        tiers=[1, 2, 3, 4, 5, 6, 7],
        strict_eval_splits=True,
    )

    assert report.counts["fen_pool_self_play"] == 1
    assert "fen_pool_self_play" not in report.required_sources
    assert "fen_pool_self_play" not in report.optional_sources


def test_absent_self_play_never_appears_as_missing_or_optional_gap():
    report = build_source_readiness_report(
        {
            "fen_pool": [{"fen": STANDARD_FEN, "is_chess960": False}],
        },
        tiers=[1, 2, 3, 4, 5, 6, 7],
        strict_eval_splits=True,
    )

    assert report.counts["fen_pool_self_play"] == 0
    issue_keys = {issue.source_key for issue in report.missing_required} | {
        issue.source_key for issue in report.optional_gaps
    }
    assert "fen_pool_self_play" not in issue_keys


def test_strict_eval_splits_require_eval_source_families():
    report = build_source_readiness_report(
        {
            "fen_pool": [{"fen": STANDARD_FEN, "is_chess960": False}],
            "puzzles": [],
            "openings": [],
            "position_evals": [],
            "best_move_evals": [],
            "endgame_positions": [],
            "mate_rows": [],
        },
        tiers=[],
        strict_eval_splits=True,
    )

    assert report.ok is False
    missing = {issue.source_key for issue in report.missing_required}
    assert {
        "fen_pool_chess960",
        "puzzles",
        "openings",
        "position_evals",
        "best_move_evals",
        "endgame_positions",
        "mate_rows",
    }.issubset(missing)
