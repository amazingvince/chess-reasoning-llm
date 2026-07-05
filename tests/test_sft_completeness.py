def test_expected_task_path_uses_tier_subdirectory(tmp_path):
    from chess_llm.sft.completeness import expected_task_path

    assert expected_task_path(tmp_path, "6.3_endgame_best_move") == (
        tmp_path / "tier6" / "6.3_endgame_best_move.jsonl"
    )


def test_audit_output_completeness_reports_missing_and_underfilled(tmp_path):
    from chess_llm.sft.completeness import audit_output_completeness

    path = tmp_path / "tier1" / "1.1_fen_to_board.jsonl"
    path.parent.mkdir()
    path.write_text('{"row": 1}\n\n', encoding="utf-8")

    issues = audit_output_completeness(
        tmp_path,
        volumes={
            "1.1_fen_to_board": 2,
            "2.1_legal_move_gen": 1,
        },
    )

    assert issues == {
        "tier1\\1.1_fen_to_board.jsonl": "underfilled: 1 / 2",
        "tier2\\2.1_legal_move_gen.jsonl": "missing expected task file (target 1)",
    }


def test_audit_output_completeness_override_applies_to_all_tasks(tmp_path):
    from chess_llm.sft.completeness import audit_output_completeness

    path = tmp_path / "tier1" / "1.1_fen_to_board.jsonl"
    path.parent.mkdir()
    path.write_text('{"row": 1}\n', encoding="utf-8")

    issues = audit_output_completeness(
        tmp_path,
        volumes={
            "1.1_fen_to_board": 100,
            "2.1_legal_move_gen": 100,
        },
        expected_volume_override=1,
    )

    assert issues == {
        "tier2\\2.1_legal_move_gen.jsonl": "missing expected task file (target 1)"
    }


def test_audit_output_completeness_can_scope_to_selected_tasks(tmp_path):
    from chess_llm.sft.completeness import audit_output_completeness

    path = tmp_path / "tier1" / "1.1_fen_to_board.jsonl"
    path.parent.mkdir()
    path.write_text('{"row": 1}\n', encoding="utf-8")

    issues = audit_output_completeness(
        tmp_path,
        volumes={
            "1.1_fen_to_board": 1,
            "2.1_legal_move_gen": 1,
        },
        task_ids=["1.1_fen_to_board"],
    )

    assert issues == {}
