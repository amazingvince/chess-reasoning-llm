# History-conditioned move-data feedback plan

## Task 1: Preserve prefix history

Files:

- `src/chess_llm/sft/sources/lichess_games.py`
- `tests/test_sft_sources_lichess.py`

Steps:

- [x] Add a failing test for per-ply `move_history`.
- [x] Store the space-separated UCI prefix before each move.
- [x] Verify the focused source test passes.

## Task 2: Add Tier 7 generator

Files:

- `src/chess_llm/sft/generators/tier7_planning.py`
- `src/chess_llm/sft/generators/__init__.py`
- `src/chess_llm/sft/settings.py`
- `src/chess_llm/sft/templates.py`
- `src/chess_llm/sft/identity.py`
- `src/chess_llm/sft/hub_upload.py`
- `src/chess_llm/sft/source_readiness.py`
- `tests/test_sft_generators.py`
- `tests/test_sft_source_readiness.py`

Steps:

- [x] Add failing tests for generated prompt, target, metadata, and registration.
- [x] Implement `7.11_history_best_move`.
- [x] Register default volume, templates, answer contract, identity fields, task
  description, and Tier 7 source readiness.
- [x] Verify the focused generator and readiness tests pass.

## Task 3: Verification and commit

Files:

- All files above plus docs.

Steps:

- [x] Run broader SFT source/generator/pipeline/readiness/completeness tests.
- [x] Run the full test suite.
- [x] Inspect staged diff.
- [x] Commit the history-conditioned move-data slice.
