"""Tests for validation/benchmark.py — frozen benchmark schema, gold answers, metrics."""

import json
import re

import chess
import pytest

from conftest import (
    STARTING_FEN, KRK_FEN, CHECKMATE_FEN, EN_PASSANT_FEN,
    PROMOTION_FEN,
)


# ── Gold answer derivation ────────────────────────────────────────────

def test_gold_board_print():
    from random import Random
    from validation.benchmark import derive_gold_answer
    from generators.base import _board_to_ascii

    raw = {"fen": STARTING_FEN}
    gold = derive_gold_answer("board_print", raw, Random(42))
    expected = _board_to_ascii(chess.Board(STARTING_FEN))
    assert gold == expected


def test_gold_board_to_fen():
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": STARTING_FEN}
    gold = derive_gold_answer("board_to_fen", raw, Random(42))
    assert gold == STARTING_FEN


def test_gold_piece_id():
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": STARTING_FEN}
    gold = derive_gold_answer("piece_id", raw, Random(42))
    # Training emits lowercase: "white pawn", "black knight", etc.
    assert "white" in gold or "black" in gold
    assert gold == gold.lower()  # no uppercase
    # Square should be stored in raw for prompt rendering
    assert "_piece_id_square" in raw


def test_gold_material_count_matches_training_format():
    """material_count gold answer matches PieceCounting generator format."""
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": STARTING_FEN}
    gold = derive_gold_answer("material_count", raw, Random(42))
    # PieceCounting generic format: "White (N pieces): ... Black (N pieces): ... Balance."
    assert "White (16 pieces):" in gold
    assert "Black (16 pieces):" in gold
    assert "Material is equal." in gold


def test_gold_material_count_imbalance():
    """Material count handles imbalanced positions (e.g. queen gone)."""
    from random import Random
    from conftest import QUEEN_GONE_FEN
    from validation.benchmark import derive_gold_answer

    raw = {"fen": QUEEN_GONE_FEN}
    gold = derive_gold_answer("material_count", raw, Random(42))
    assert "Black is up" in gold


def test_gold_state_tracking():
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": STARTING_FEN}
    gold = derive_gold_answer("state_tracking", raw, Random(42))
    # Gold is a valid FEN, moves are stored in raw
    chess.Board(gold)  # should not raise
    assert "_state_tracking_moves" in raw
    assert raw["_state_tracking_moves"]  # non-empty


def test_gold_legality_check():
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": STARTING_FEN}
    gold = derive_gold_answer("legality_check", raw, Random(42))
    assert gold in ("Yes, the move is legal.", "No, the move is not legal.")
    assert "_legality_check_move" in raw


def test_gold_legal_moves():
    from random import Random
    from validation.benchmark import derive_gold_answer
    from validation.eval_harness import answer_legal_moves

    raw = {"fen": STARTING_FEN}
    gold = derive_gold_answer("legal_moves", raw, Random(42))
    assert gold == answer_legal_moves(STARTING_FEN)


def test_gold_check_detection_matches_training_format():
    """check_detection gold matches CheckDetection generator format."""
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": CHECKMATE_FEN}
    gold = derive_gold_answer("check_detection", raw, Random(42))
    assert gold == "Checkmate."  # training format has period

    raw_normal = {"fen": STARTING_FEN}
    gold_normal = derive_gold_answer("check_detection", raw_normal, Random(42))
    assert gold_normal == "Normal position -- no check, checkmate, or stalemate."


def test_gold_captures_matches_training_format():
    """captures gold matches AvailableCaptures generator format."""
    from random import Random
    from validation.benchmark import derive_gold_answer

    # Starting position has no captures
    raw = {"fen": STARTING_FEN}
    gold = derive_gold_answer("captures", raw, Random(42))
    assert gold == "No captures available."

    # Position after 1. e4 d5 has captures
    fen_e4d5 = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    raw2 = {"fen": fen_e4d5}
    gold2 = derive_gold_answer("captures", raw2, Random(42))
    assert "e4d5" in gold2


def test_gold_special_rules_matches_training_format():
    """special_rules gold answer matches SpecialRules generator format."""
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": EN_PASSANT_FEN}
    gold = derive_gold_answer("special_rules", raw, Random(42))
    # SpecialRules format uses "Castling available:" / "En passant possible:"
    assert "En passant possible:" in gold
    assert "Castling available:" in gold


def test_gold_special_rules_no_specials():
    """Position with no special moves returns proper message."""
    from random import Random
    from validation.benchmark import derive_gold_answer

    # KRK has no castling, en passant, or promotion
    raw = {"fen": KRK_FEN}
    gold = derive_gold_answer("special_rules", raw, Random(42))
    assert gold == "No special moves available."


def test_gold_hanging_pieces_matches_training_format():
    """hanging_pieces gold answer matches HangingPieces generator format."""
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": STARTING_FEN}
    gold = derive_gold_answer("hanging_pieces", raw, Random(42))
    # Training format: "Hanging pieces: ..." or "No hanging pieces —..."
    assert gold.startswith("Hanging pieces:") or gold.startswith("No hanging pieces")


def test_gold_threats():
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": STARTING_FEN}
    gold = derive_gold_answer("threats", raw, Random(42))
    # Training format: "{Color} threatens: ..." or "{Color} has no immediate threats."
    assert "threatens:" in gold or "has no immediate threats" in gold
    assert "_threats_color" in raw


def test_gold_tactical_patterns():
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": STARTING_FEN, "themes": ["fork", "pin"],
           "solution_first_move": "e2e4"}
    gold = derive_gold_answer("tactical_patterns", raw, Random(42))
    assert gold == "The tactic is fork, pin. Best move: e2e4"


def test_gold_eval_bucket():
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": STARTING_FEN, "cp": 30, "mate": None}
    gold = derive_gold_answer("eval_bucket", raw, Random(42))
    assert "equal" in gold.lower()


def test_gold_endgame_wdl():
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": KRK_FEN, "wdl": 2}
    gold = derive_gold_answer("endgame_wdl", raw, Random(42))
    assert "Win" in gold


def test_gold_opening_name():
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": STARTING_FEN, "name": "Sicilian Defense", "eco": "B20"}
    gold = derive_gold_answer("opening_name", raw, Random(42))
    assert gold == "Sicilian Defense (ECO: B20)"


def test_gold_opening_continuation():
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": STARTING_FEN,
           "book_moves": [("e2e4", 100), ("d2d4", 80), ("g1f3", 20)]}
    gold = derive_gold_answer("opening_continuation", raw, Random(42))
    assert gold.startswith("Top continuations:")
    assert "e2e4" in gold


def test_gold_opening_continuation_no_data():
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": STARTING_FEN}
    gold = derive_gold_answer("opening_continuation", raw, Random(42))
    assert gold == ""


def test_gold_castling_rules_960():
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": STARTING_FEN}
    gold = derive_gold_answer("castling_rules_960", raw, Random(42))
    assert "Castling available:" in gold


def test_gold_binary_choice():
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": STARTING_FEN, "better_move": "e2e4",
           "move_a": "e2e4", "move_b": "d2d4"}
    gold = derive_gold_answer("binary_choice", raw, Random(42))
    assert gold == "e2e4"


def test_gold_best_move_from_engine_eval():
    """best_move task uses engine-labeled best_move field."""
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": STARTING_FEN, "best_move": "d2d4", "cp": 35}
    gold = derive_gold_answer("best_move", raw, Random(42))
    assert gold == "d2d4"


def test_gold_best_move_empty_without_label():
    """best_move returns empty when no engine label is present."""
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": STARTING_FEN}
    gold = derive_gold_answer("best_move", raw, Random(42))
    assert gold == ""


def test_freeze_planning_engine_labeled():
    """Planning split with engine-labeled evals + puzzles produces examples."""
    from validation.benchmark import freeze_split

    raw = [
        {"fen": STARTING_FEN, "best_move": "d2d4", "cp": 35},
        {"fen": STARTING_FEN, "puzzle_id": "p1", "solution_first_move": "e2e4"},
    ]
    examples = freeze_split("planning", raw, seed=42)
    assert len(examples) == 2
    assert examples[0].task_type == "best_move"
    assert examples[0].gold_answer == "d2d4"
    assert examples[1].task_type == "puzzle_solve"
    assert examples[1].gold_answer == "e2e4"


def test_gold_puzzle_solve():
    from random import Random
    from validation.benchmark import derive_gold_answer

    raw = {"fen": STARTING_FEN, "solution_first_move": "e2e4"}
    gold = derive_gold_answer("puzzle_solve", raw, Random(42))
    assert gold == "e2e4"


# ── Metric functions ──────────────────────────────────────────────────


def test_exact_match_hit():
    from validation.benchmark import exact_match
    assert exact_match("Normal", "Normal") == 1.0


def test_exact_match_miss():
    from validation.benchmark import exact_match
    assert exact_match("Normal", "Check") == 0.0


def test_exact_match_whitespace():
    from validation.benchmark import exact_match
    assert exact_match("  Normal  ", "normal") == 1.0


def test_jaccard_identical():
    from validation.benchmark import jaccard_similarity
    assert jaccard_similarity("a b c", "a b c") == 1.0


def test_jaccard_partial():
    from validation.benchmark import jaccard_similarity
    score = jaccard_similarity("a b c", "a b d")
    assert 0.0 < score < 1.0
    assert abs(score - 2.0 / 4.0) < 1e-9  # |{a,b}| / |{a,b,c,d}|


def test_jaccard_disjoint():
    from validation.benchmark import jaccard_similarity
    assert jaccard_similarity("a b", "c d") == 0.0


def test_eval_bucket_match():
    from validation.benchmark import eval_bucket_accuracy
    assert eval_bucket_accuracy(
        "The position is roughly equal.",
        "Position is equal."
    ) == 1.0


def test_eval_bucket_mismatch():
    from validation.benchmark import eval_bucket_accuracy
    assert eval_bucket_accuracy(
        "White has a decisive advantage.",
        "Position is equal."
    ) == 0.0


def test_format_compliance_valid():
    from validation.benchmark import format_compliance
    text = "<think>I see a tactic.</think><move>e2e4</move>"
    assert format_compliance(text) == 1.0


def test_format_compliance_invalid():
    from validation.benchmark import format_compliance
    assert format_compliance("just e2e4") == 0.0


def test_legal_move_rate_legal():
    from validation.benchmark import legal_move_rate
    text = "<think>Best move.</think><move>e2e4</move>"
    assert legal_move_rate(text, STARTING_FEN) == 1.0


def test_legal_move_rate_illegal():
    from validation.benchmark import legal_move_rate
    text = "<think>Bad move.</think><move>e1e5</move>"
    assert legal_move_rate(text, STARTING_FEN) == 0.0


def test_pass_at_k_hit():
    from validation.benchmark import pass_at_k
    preds = [
        "<think>no</think><move>d2d4</move>",
        "<think>yes</think><move>e2e4</move>",
    ]
    assert pass_at_k(preds, "e2e4") == 1.0


def test_pass_at_k_miss():
    from validation.benchmark import pass_at_k
    preds = [
        "<think>no</think><move>d2d4</move>",
        "<think>also no</think><move>a2a3</move>",
    ]
    assert pass_at_k(preds, "e2e4") == 0.0


def test_threat_f1_perfect():
    from validation.benchmark import threat_f1
    gold = "White threatens: knight on e4, pawn on d5."
    pred = "White threatens: knight on e4, pawn on d5."
    assert threat_f1(pred, gold) == 1.0


def test_threat_f1_partial():
    from validation.benchmark import threat_f1
    gold = "White threatens: knight on e4, pawn on d5."
    pred = "White threatens: knight on e4."
    score = threat_f1(pred, gold)
    # precision=1, recall=0.5 -> F1=2/3
    assert abs(score - 2.0 / 3.0) < 1e-9


def test_threat_f1_no_threats():
    from validation.benchmark import threat_f1
    gold = "White has no immediate threats."
    pred = "White has no immediate threats."
    assert threat_f1(pred, gold) == 1.0


def test_threat_f1_miss():
    from validation.benchmark import threat_f1
    gold = "White threatens: knight on e4."
    pred = "White has no immediate threats."
    assert threat_f1(pred, gold) == 0.0


def test_continuation_rank_top():
    from validation.benchmark import continuation_rank
    gold = "Top continuations: e2e4 (50%), d2d4 (30%), g1f3 (20%)."
    pred = "The best continuation is e2e4."
    assert continuation_rank(pred, gold) == 1.0


def test_continuation_rank_second():
    from validation.benchmark import continuation_rank
    gold = "Top continuations: e2e4 (50%), d2d4 (30%), g1f3 (20%)."
    pred = "I would play d2d4 here."
    assert continuation_rank(pred, gold) == 0.5


def test_continuation_rank_miss():
    from validation.benchmark import continuation_rank
    gold = "Top continuations: e2e4 (50%), d2d4 (30%), g1f3 (20%)."
    pred = "I suggest a2a3."
    assert continuation_rank(pred, gold) == 0.0


def test_centipawn_loss_zero():
    from validation.benchmark import centipawn_loss
    assert centipawn_loss(100.0, 100.0) == 0.0


def test_centipawn_loss_positive():
    from validation.benchmark import centipawn_loss
    assert centipawn_loss(200.0, 150.0) == 50.0


def test_centipawn_loss_better_than_gold():
    """If predicted_cp > gold_cp, loss is 0 (clamped)."""
    from validation.benchmark import centipawn_loss
    assert centipawn_loss(100.0, 200.0) == 0.0


# ── Freeze/score integration ─────────────────────────────────────────


def test_freeze_split_count():
    from validation.benchmark import freeze_split

    raw = [{"fen": STARTING_FEN}] * 10
    examples = freeze_split("perception", raw, seed=42)
    assert len(examples) == 10


def test_freeze_split_round_robin_perception():
    from validation.benchmark import freeze_split

    raw = [{"fen": STARTING_FEN}] * 10
    examples = freeze_split("perception", raw, seed=42)
    task_types = [ex.task_type for ex in examples]
    # perception: board_print, board_to_fen, piece_id, material_count, state_tracking, ...
    assert task_types[0] == "board_print"
    assert task_types[1] == "board_to_fen"
    assert task_types[2] == "piece_id"
    assert task_types[3] == "material_count"
    assert task_types[4] == "state_tracking"
    assert task_types[5] == "board_print"  # wraps around


def test_freeze_split_round_robin_rules():
    from validation.benchmark import freeze_split

    raw = [{"fen": STARTING_FEN}] * 10
    examples = freeze_split("rules", raw, seed=42)
    task_types = [ex.task_type for ex in examples]
    # rules: legal_moves, check_detection, captures, special_rules, legality_check, ...
    assert task_types[0] == "legal_moves"
    assert task_types[1] == "check_detection"
    assert task_types[2] == "captures"
    assert task_types[3] == "special_rules"
    assert task_types[4] == "legality_check"


def test_freeze_split_round_robin_tactics():
    from validation.benchmark import freeze_split

    raw = [
        {"fen": STARTING_FEN, "themes": ["fork"], "solution_first_move": "e2e4"}
        for _ in range(8)
    ]
    examples = freeze_split("tactics", raw, seed=42)
    task_types = [ex.task_type for ex in examples]
    # tactics: capture_id, hanging_pieces, threats, tactical_patterns, ...
    assert task_types[0] == "capture_id"
    assert task_types[1] == "hanging_pieces"
    assert task_types[2] == "threats"
    assert task_types[3] == "tactical_patterns"


def test_freeze_split_round_robin_chess960():
    from validation.benchmark import freeze_split

    raw = [{"fen": STARTING_FEN, "is_chess960": True}] * 6
    examples = freeze_split("chess960", raw, seed=42)
    task_types = [ex.task_type for ex in examples]
    # chess960: legal_moves_960, check_detection_960, castling_rules_960, ...
    assert task_types[0] == "legal_moves_960"
    assert task_types[1] == "check_detection_960"
    assert task_types[2] == "castling_rules_960"


def test_freeze_split_deterministic():
    from validation.benchmark import freeze_split

    raw = [{"fen": STARTING_FEN}, {"fen": KRK_FEN}]
    e1 = freeze_split("rules", raw, seed=99)
    e2 = freeze_split("rules", raw, seed=99)
    assert [ex.gold_answer for ex in e1] == [ex.gold_answer for ex in e2]


def test_freeze_split_no_unfilled_placeholders():
    from validation.benchmark import freeze_split

    raw = [{"fen": STARTING_FEN}] * 5
    examples = freeze_split("rules", raw, seed=42)
    placeholder_re = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")
    for ex in examples:
        assert not placeholder_re.search(ex.prompt), (
            f"Unfilled placeholder in {ex.task_type} prompt: {ex.prompt!r}"
        )


def test_freeze_split_no_unfilled_placeholders_perception():
    from validation.benchmark import freeze_split

    raw = [{"fen": STARTING_FEN}] * 5
    examples = freeze_split("perception", raw, seed=42)
    placeholder_re = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")
    for ex in examples:
        assert not placeholder_re.search(ex.prompt), (
            f"Unfilled placeholder in {ex.task_type} prompt: {ex.prompt!r}"
        )


def test_freeze_planning_source_detection():
    from validation.benchmark import freeze_split

    raw_puzzle = [{"fen": STARTING_FEN, "puzzle_id": "abc123",
                   "solution_first_move": "e2e4"}]
    raw_bestmove = [{"fen": KRK_FEN, "best_move": "h1h5"}]
    examples = freeze_split("planning", raw_puzzle + raw_bestmove, seed=42)
    assert examples[0].task_type == "puzzle_solve"
    assert examples[1].task_type == "best_move"


def test_freeze_fallback_on_empty_gold():
    """When primary task has no gold, fallback to another task type."""
    from validation.benchmark import freeze_split

    # Two opening rows without book_moves.
    # Round-robin assigns [opening_name, opening_continuation].
    # Row 0 → opening_name (has name/eco → gold OK).
    # Row 1 → opening_continuation (no book_moves → gold empty)
    #       → fallback to opening_name → gold OK.
    raw = [{"fen": STARTING_FEN, "name": "Test", "eco": "A00"},
           {"fen": STARTING_FEN, "name": "Test2", "eco": "A01"}]
    examples = freeze_split("openings", raw, seed=42)
    assert len(examples) == 2  # neither dropped
    # Both should have a valid gold answer
    for ex in examples:
        assert ex.gold_answer


def test_freeze_logs_warning_on_underfill(caplog):
    """When all task types fail for a row, it is skipped and a warning logs."""
    import logging
    from validation.benchmark import freeze_split

    # planning split: source-detected.  A row with neither puzzle_id
    # nor best_move nor solution_first_move → empty gold for both tasks.
    raw = [{"fen": STARTING_FEN}]
    with caplog.at_level(logging.WARNING, logger="validation.benchmark"):
        examples = freeze_split("planning", raw, seed=42)
    assert len(examples) == 0
    assert any("skipped" in r.message for r in caplog.records)


def test_freeze_warns_task_mix_drift(caplog):
    """When a task type gets 0 examples due to fallback, warn about drift."""
    import logging
    from validation.benchmark import freeze_split

    # All opening rows lack book_moves, so opening_continuation always
    # falls back to opening_name. That means opening_continuation has
    # 0 coverage — the warning should fire.
    raw = [
        {"fen": STARTING_FEN, "name": f"Op{i}", "eco": f"A{i:02d}"}
        for i in range(4)
    ]
    with caplog.at_level(logging.WARNING, logger="validation.benchmark"):
        examples = freeze_split("openings", raw, seed=42)
    assert len(examples) == 4  # all preserved via fallback
    assert any("opening_continuation" in r.message and "0 examples" in r.message
               for r in caplog.records)


def test_score_prediction_dispatch():
    from validation.benchmark import BenchmarkExample, score_prediction

    ex = BenchmarkExample(
        example_id="rules_00000", split="rules", task_type="legal_moves",
        fen=STARTING_FEN, prompt="...", gold_answer="a2a3 a2a4",
        metric_type="jaccard", metadata={},
    )
    scores = score_prediction(ex, "a2a3 a2a4")
    assert scores["primary"] == 1.0


def test_score_split_aggregation():
    from validation.benchmark import BenchmarkExample, score_split

    examples = [
        BenchmarkExample(
            example_id="test_00000", split="test", task_type="check_detection",
            fen=STARTING_FEN, prompt="...", gold_answer="Normal",
            metric_type="exact_match", metadata={},
        ),
        BenchmarkExample(
            example_id="test_00001", split="test", task_type="check_detection",
            fen=CHECKMATE_FEN, prompt="...", gold_answer="Checkmate",
            metric_type="exact_match", metadata={},
        ),
    ]
    preds = {"test_00000": "Normal", "test_00001": "Checkmate"}
    metrics = score_split(examples, preds)
    assert metrics["check_detection"] == 1.0
    assert metrics["overall"] == 1.0


def test_score_split_with_acpl():
    """ACPL scores are aggregated when provided."""
    from validation.benchmark import BenchmarkExample, score_split

    examples = [
        BenchmarkExample(
            example_id="plan_00000", split="planning", task_type="best_move",
            fen=STARTING_FEN, prompt="...", gold_answer="e2e4",
            metric_type="move_extraction", metadata={"cp": 50},
        ),
        BenchmarkExample(
            example_id="plan_00001", split="planning", task_type="best_move",
            fen=STARTING_FEN, prompt="...", gold_answer="d2d4",
            metric_type="move_extraction", metadata={"cp": 40},
        ),
    ]
    preds = {
        "plan_00000": "<think>reason</think><move>e2e4</move>",
        "plan_00001": "<think>reason</think><move>d2d4</move>",
    }
    acpl = {"plan_00000": 10.0, "plan_00001": 20.0}
    metrics = score_split(examples, preds, acpl_scores=acpl)
    assert "acpl" in metrics
    assert metrics["acpl"] == 15.0  # mean(10, 20)
    assert "best_move_acpl" in metrics


def test_json_serialization_roundtrip(tmp_path):
    from validation.benchmark import (
        BenchmarkExample, save_benchmark, load_benchmark,
    )

    examples = [
        BenchmarkExample(
            example_id="perc_00000", split="perception", task_type="board_print",
            fen=STARTING_FEN, prompt="Show me the board.",
            gold_answer="8 r n b q k b n r\n...",
            metric_type="exact_match", metadata={"source": "test"},
        ),
    ]
    path = tmp_path / "test.jsonl"
    save_benchmark(examples, str(path))
    loaded = load_benchmark(str(path))
    assert len(loaded) == 1
    assert loaded[0].example_id == "perc_00000"
    assert loaded[0].metadata == {"source": "test"}


# ── Move extraction metric (best_move / puzzle_solve) ────────────────


def test_move_extraction_tagged_correct():
    """Tier 7 <think>/<move> output scores 1.0 when move matches gold."""
    from validation.benchmark import move_extraction_match
    pred = "<think>I see a tactic.</think><move>e2e4</move>"
    assert move_extraction_match(pred, "e2e4") == 1.0


def test_move_extraction_tagged_wrong():
    """Tier 7 <think>/<move> output scores 0.0 when move doesn't match."""
    from validation.benchmark import move_extraction_match
    pred = "<think>I see a tactic.</think><move>d2d4</move>"
    assert move_extraction_match(pred, "e2e4") == 0.0


def test_move_extraction_bare_uci():
    """Bare UCI (no tags) falls back to exact match."""
    from validation.benchmark import move_extraction_match
    assert move_extraction_match("e2e4", "e2e4") == 1.0
    assert move_extraction_match("d2d4", "e2e4") == 0.0


def test_score_prediction_best_move_with_tags():
    """score_prediction for best_move accepts <think>/<move> format."""
    from validation.benchmark import BenchmarkExample, score_prediction

    ex = BenchmarkExample(
        example_id="plan_00000", split="planning", task_type="best_move",
        fen=STARTING_FEN, prompt="...", gold_answer="e2e4",
        metric_type="move_extraction", metadata={},
    )
    scores = score_prediction(ex, "<think>reason</think><move>e2e4</move>")
    assert scores["primary"] == 1.0
    assert scores["format_compliance"] == 1.0
    assert scores["legal_move"] == 1.0


def test_score_prediction_puzzle_solve_with_tags():
    """score_prediction for puzzle_solve accepts <think>/<move> format."""
    from validation.benchmark import BenchmarkExample, score_prediction

    ex = BenchmarkExample(
        example_id="plan_00001", split="planning", task_type="puzzle_solve",
        fen=STARTING_FEN, prompt="...", gold_answer="e2e4",
        metric_type="move_extraction", metadata={},
    )
    scores = score_prediction(ex, "<think>Nf3 blocks.</think><move>e2e4</move>")
    assert scores["primary"] == 1.0


# ── Endgame best_move: no secondary format/legal metrics ─────────────


def test_score_prediction_endgame_best_move_no_secondary():
    """endgame_best_move should NOT get format_compliance/legal_move."""
    from validation.benchmark import BenchmarkExample, score_prediction

    ex = BenchmarkExample(
        example_id="end_00000", split="endgames", task_type="endgame_best_move",
        fen=KRK_FEN, prompt="...", gold_answer="h1h7",
        metric_type="exact_match", metadata={},
    )
    scores = score_prediction(ex, "h1h7")
    assert scores["primary"] == 1.0
    assert "format_compliance" not in scores
    assert "legal_move" not in scores


# ── Binary choice: move_choice metric ────────────────────────────────


def test_move_choice_bare():
    """Bare move matches gold (no candidates)."""
    from validation.benchmark import move_choice_match
    assert move_choice_match("e2e4", "e2e4") == 1.0


def test_move_choice_with_explanation():
    """Move followed by explanation still scores 1.0."""
    from validation.benchmark import move_choice_match
    assert move_choice_match(
        "e2e4 because it controls the center", "e2e4",
        candidates=("e2e4", "d2d4"),
    ) == 1.0


def test_move_choice_wrong():
    """Different move scores 0.0."""
    from validation.benchmark import move_choice_match
    assert move_choice_match(
        "d2d4 because it controls the center", "e2e4",
        candidates=("e2e4", "d2d4"),
    ) == 0.0


def test_move_choice_rejects_mentioned_gold():
    """Gold move mentioned only in explanation (not as choice) scores 0.0."""
    from validation.benchmark import move_choice_match
    # Model chose d2d4 but explains "better than e2e4" — gold e2e4 is
    # the rejected option, not the chosen one.
    assert move_choice_match(
        "d2d4 is better than e2e4", "e2e4",
        candidates=("e2e4", "d2d4"),
    ) == 0.0


def test_move_choice_restates_both_then_chooses():
    """Model restates both candidates before choosing — correct one scored."""
    from validation.benchmark import move_choice_match
    assert move_choice_match(
        "Between e2e4 and d2d4, d2d4 is better", "d2d4",
        candidates=("e2e4", "d2d4"),
    ) == 1.0
    # Same text but gold is e2e4 -> should fail
    assert move_choice_match(
        "Between e2e4 and d2d4, d2d4 is better", "e2e4",
        candidates=("e2e4", "d2d4"),
    ) == 0.0


def test_move_choice_negation():
    """'I choose d2d4, not e2e4' picks d2d4."""
    from validation.benchmark import move_choice_match
    assert move_choice_match(
        "I choose d2d4, not e2e4", "d2d4",
        candidates=("e2e4", "d2d4"),
    ) == 1.0


def test_move_choice_over_with_repeated_other():
    """'I choose d2d4 over e2e4 because e2e4...e2e4' picks d2d4."""
    from validation.benchmark import move_choice_match
    assert move_choice_match(
        "I choose d2d4 over e2e4 because e2e4 weakens the center and e2e4 is passive",
        "d2d4",
        candidates=("e2e4", "d2d4"),
    ) == 1.0


def test_move_choice_not_first():
    """'Not e2e4; choose d2d4' picks d2d4."""
    from validation.benchmark import move_choice_match
    assert move_choice_match(
        "Not e2e4; choose d2d4", "d2d4",
        candidates=("e2e4", "d2d4"),
    ) == 1.0


def test_move_choice_is_not_best():
    """'e2e4 is not best; d2d4 is better' picks d2d4."""
    from validation.benchmark import move_choice_match
    assert move_choice_match(
        "e2e4 is not best; d2d4 is better", "d2d4",
        candidates=("e2e4", "d2d4"),
    ) == 1.0


def test_move_choice_rather_than():
    """'Rather than e2e4, play d2d4' picks d2d4."""
    from validation.benchmark import move_choice_match
    assert move_choice_match(
        "Rather than e2e4, play d2d4", "d2d4",
        candidates=("e2e4", "d2d4"),
    ) == 1.0


def test_move_choice_avoid():
    """'Avoid e2e4, d2d4 is stronger' picks d2d4."""
    from validation.benchmark import move_choice_match
    assert move_choice_match(
        "Avoid e2e4, d2d4 is stronger", "d2d4",
        candidates=("e2e4", "d2d4"),
    ) == 1.0


def test_score_prediction_binary_choice_with_explanation():
    """Binary choice accepts answers with explanations."""
    from validation.benchmark import BenchmarkExample, score_prediction

    ex = BenchmarkExample(
        example_id="mate_00000", split="mate", task_type="binary_choice",
        fen=STARTING_FEN, prompt="...", gold_answer="e2e4",
        metric_type="move_choice",
        metadata={"move_a": "e2e4", "move_b": "d2d4", "better_move": "e2e4"},
    )
    scores = score_prediction(ex, "e2e4 because it controls the center")
    assert scores["primary"] == 1.0


def test_score_prediction_binary_choice_restated():
    """Binary choice with both candidates restated scores correctly."""
    from validation.benchmark import BenchmarkExample, score_prediction

    ex = BenchmarkExample(
        example_id="mate_00001", split="mate", task_type="binary_choice",
        fen=STARTING_FEN, prompt="...", gold_answer="d2d4",
        metric_type="move_choice",
        metadata={"move_a": "e2e4", "move_b": "d2d4", "better_move": "d2d4"},
    )
    scores = score_prediction(ex, "Between e2e4 and d2d4, d2d4 is better")
    assert scores["primary"] == 1.0


# ── pass_at_k consistency with bare UCI ──────────────────────────────


def test_pass_at_k_bare_uci():
    """pass_at_k matches bare UCI predictions (no <move> tag)."""
    from validation.benchmark import pass_at_k
    assert pass_at_k(["e2e4"], "e2e4") == 1.0


def test_pass_at_k_bare_uci_miss():
    """pass_at_k rejects wrong bare UCI."""
    from validation.benchmark import pass_at_k
    assert pass_at_k(["d2d4"], "e2e4") == 0.0


def test_pass_at_k_mixed_formats():
    """pass_at_k finds gold in mix of tagged and bare predictions."""
    from validation.benchmark import pass_at_k
    preds = [
        "<think>no</think><move>d2d4</move>",
        "e2e4",
    ]
    assert pass_at_k(preds, "e2e4") == 1.0


# ── legal_move_rate denominator ──────────────────────────────────────


def test_legal_move_rate_no_tag_returns_none():
    """Bare UCI without <move> tag returns None, not 0.0."""
    from validation.benchmark import legal_move_rate
    assert legal_move_rate("e2e4", STARTING_FEN) is None


def test_legal_move_rate_none_excluded_from_aggregate():
    """score_split excludes None legal_move from the denominator."""
    from validation.benchmark import BenchmarkExample, score_split

    examples = [
        BenchmarkExample(
            example_id="plan_00000", split="planning", task_type="best_move",
            fen=STARTING_FEN, prompt="...", gold_answer="e2e4",
            metric_type="move_extraction", metadata={},
        ),
        BenchmarkExample(
            example_id="plan_00001", split="planning", task_type="best_move",
            fen=STARTING_FEN, prompt="...", gold_answer="d2d4",
            metric_type="move_extraction", metadata={},
        ),
    ]
    preds = {
        # Tagged: legal move
        "plan_00000": "<think>yes</think><move>e2e4</move>",
        # Bare UCI: no tag, so legal_move should be None (excluded)
        "plan_00001": "d2d4",
    }
    metrics = score_split(examples, preds)
    # Only one example has a tag -> denominator is 1, and that move is legal
    assert metrics["best_move_legal_move"] == 1.0


# ── Oracle validation ────────────────────────────────────────────────


def test_oracle_all_splits():
    """Gold answers must self-score 1.0 on every task type in every split.

    This covers the plan's '100% accuracy on oracle' requirement by
    freezing synthetic examples for all 9 splits and verifying that
    feeding each gold_answer as the prediction yields primary=1.0.
    """
    from validation.benchmark import freeze_split, validate_oracle

    fen = STARTING_FEN
    # Build raw data that covers every split
    raw_by_split = {
        "perception": [{"fen": fen}] * 5,
        "rules": [{"fen": fen}] * 5,
        "tactics": [
            {"fen": fen, "themes": ["fork"], "solution_first_move": "e2e4"}
            for _ in range(4)
        ],
        "evaluation": [{"fen": fen, "cp": 30, "mate": None}] * 3,
        "openings": [
            {"fen": fen, "name": f"Op{i}", "eco": f"A{i:02d}",
             "book_moves": [("e2e4", 100), ("d2d4", 80)]}
            for i in range(2)
        ],
        "endgames": [
            {"fen": KRK_FEN, "wdl": 2, "material": "KRK"}
            for _ in range(3)
        ],
        "planning": [
            {"fen": fen, "best_move": "d2d4", "cp": 35},
            {"fen": fen, "puzzle_id": "p1", "solution_first_move": "e2e4"},
        ],
        "chess960": [{"fen": fen, "is_chess960": True}] * 3,
        "mate": [
            {"fen": fen, "move_a": "e2e4", "move_b": "d2d4", "better_move": "e2e4"}
            for _ in range(2)
        ],
    }

    all_failures: list[str] = []
    for split_name, raw in raw_by_split.items():
        examples = freeze_split(split_name, raw, seed=42)
        assert examples, f"No examples frozen for {split_name}"
        failures = validate_oracle(examples)
        all_failures.extend(failures)

    assert not all_failures, (
        f"Oracle failures ({len(all_failures)}):\n"
        + "\n".join(all_failures[:20])
    )


def test_partial_refreeze_uses_manifest_seed(tmp_path):
    """Partial refreeze uses the manifest's seed, not the CLI argument.

    Seed-dependent golds (e.g. legality_check) must be identical whether
    produced by a full freeze or a partial refreeze of the same split.
    """
    from validation.benchmark import freeze_and_save, load_benchmark

    fen = STARTING_FEN
    splits = {
        "rules": [{"fen": fen}] * 10,
        "perception": [{"fen": fen}] * 5,
    }

    # Full freeze with seed=42
    freeze_and_save(splits, str(tmp_path), seed=42, version="v1")
    full_rules = load_benchmark(str(tmp_path / "rules.jsonl"))

    # Partial refreeze of rules with a DIFFERENT seed arg
    freeze_and_save(
        {"rules": [{"fen": fen}] * 10},
        str(tmp_path), seed=999, version="v2", clean=False,
    )
    partial_rules = load_benchmark(str(tmp_path / "rules.jsonl"))

    # Gold answers must match — partial used the manifest's seed=42
    full_golds = [ex.gold_answer for ex in full_rules]
    partial_golds = [ex.gold_answer for ex in partial_rules]
    assert full_golds == partial_golds, (
        "Partial refreeze produced different golds — seed mismatch"
    )

    # Manifest should still say seed=42, version=v1
    import json
    with open(tmp_path / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["seed"] == 42
    assert manifest["version"] == "v1"


# ── Dismissal-punctuation patterns ───────────────────────────────────


def test_move_choice_question_no():
    """'e2e4? No, d2d4 is stronger.' picks d2d4."""
    from validation.benchmark import move_choice_match
    assert move_choice_match(
        "e2e4? No, d2d4 is stronger.", "d2d4",
        candidates=("e2e4", "d2d4"),
    ) == 1.0


def test_move_choice_question_no_period():
    """'e2e4? No. d2d4.' picks d2d4."""
    from validation.benchmark import move_choice_match
    assert move_choice_match(
        "e2e4? No. d2d4.", "d2d4",
        candidates=("e2e4", "d2d4"),
    ) == 1.0


# ── freeze_and_save coverage enforcement ─────────────────────────────


def test_freeze_and_save_rejects_coverage_gaps(tmp_path):
    """freeze_and_save raises on zero-coverage task types."""
    from validation.benchmark import freeze_and_save

    # Openings without book_moves -> opening_continuation has 0 coverage
    splits = {
        "openings": [
            {"fen": STARTING_FEN, "name": f"Op{i}", "eco": f"A{i:02d}"}
            for i in range(4)
        ],
    }
    with pytest.raises(ValueError, match="Task coverage gaps"):
        freeze_and_save(splits, str(tmp_path), seed=42)
