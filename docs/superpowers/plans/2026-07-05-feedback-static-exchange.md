# Static exchange feedback plan

## Task 1: Red tests

Files:

- `tests/test_sft_generators.py`

Steps:

- [x] Add a failing test for fixed static-exchange sequence output.
- [x] Add a failing registration test for default volume, answer contract,
  identity, and task description.

## Task 2: Generator and metadata

Files:

- `src/chess_llm/sft/generators/tier3_tactics.py`
- `src/chess_llm/sft/generators/__init__.py`
- `src/chess_llm/sft/settings.py`
- `src/chess_llm/sft/templates.py`
- `src/chess_llm/sft/identity.py`
- `src/chess_llm/sft/hub_upload.py`

Steps:

- [x] Implement legal-capture selection over `fen_pool`.
- [x] Compute least-valuable-attacker recapture sequences.
- [x] Emit fixed grammar and metadata.
- [x] Register the task in Tier 3 defaults and public metadata surfaces.

## Task 3: Verification and commit

Files:

- All files above plus docs.

Steps:

- [x] Run focused static-exchange tests.
- [x] Run broader SFT generator/completeness tests.
- [x] Run the full test suite.
- [x] Inspect staged diff.
- [x] Commit the static-exchange slice.
