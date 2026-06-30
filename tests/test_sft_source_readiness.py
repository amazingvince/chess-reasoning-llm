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
            "mate_rows": [],
        },
        tiers=[7],
        strict_eval_splits=False,
    )

    assert report.ok is True
    assert [issue.source_key for issue in report.optional_gaps] == ["mate_rows"]
    assert report.to_dict()["optional_gaps"][0]["required_by"] == [
        "tier7_optional"
    ]


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
