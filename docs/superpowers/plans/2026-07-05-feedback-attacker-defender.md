# Attacker/defender feedback plan

## Task 1: Red tests

Files:

- `tests/test_sft_generators.py`

Steps:

- [x] Add a failing test for fixed attacker/defender count grammar.
- [x] Add a failing registration test for answer contract and query-square identity.

## Task 2: Fixed grammar and metadata

Files:

- `src/chess_llm/sft/generators/tier3_tactics.py`
- `src/chess_llm/sft/templates.py`
- `src/chess_llm/sft/identity.py`
- `src/chess_llm/sft/hub_upload.py`

Steps:

- [x] Emit five fixed answer lines.
- [x] Include white/black attacker counts and defender count.
- [x] Preserve query square and count/list metadata.
- [x] Add answer contract and identity field.

## Task 3: Verification and commit

Files:

- All files above plus docs.

Steps:

- [x] Run focused attacker/defender tests.
- [x] Run broader SFT generator tests.
- [x] Run the full test suite.
- [x] Inspect staged diff.
- [x] Commit the attacker/defender slice.
