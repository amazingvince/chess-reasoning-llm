"""Shared fixtures and constants for the chess SFT test suite.

Environment isolation runs at import time — before any project module
touches ``config/settings.py`` — so HF_HOME and OUTPUT never hit E:.
"""

import os
import sys
import tempfile

# ── Environment isolation ────────────────────────────────────────────
# Must execute BEFORE any project import (config/settings.py uses
# os.environ.setdefault, so pre-setting wins).
_TMP = tempfile.mkdtemp(prefix="chess_sft_test_")
os.environ["HF_HOME"] = _TMP
os.environ["CHESS_SFT_OUTPUT"] = os.path.join(_TMP, "output")

# Ensure sft/make_data/ is on sys.path (pytest.ini pythonpath=. handles
# this, but belt-and-suspenders for IDE runners).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_SRC_ROOT = os.path.abspath(os.path.join(_PROJECT_ROOT, "..", "..", "src"))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

# Also add tests/ dir so test files can ``from conftest import ...``.
_TESTS_DIR = os.path.dirname(__file__)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Mock the ``datasets`` package when it is not installed or broken so that
# ``from datasets import load_dataset`` in sources/lichess_evals.py
# does not crash at import time.
try:
    import datasets  # noqa: F401
except Exception:
    from unittest.mock import MagicMock

    sys.modules["datasets"] = MagicMock()

# ── Reusable FEN constants ───────────────────────────────────────────
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
PROMOTION_FEN = "8/P7/8/8/8/8/8/4K2k w - - 0 1"
EN_PASSANT_FEN = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
KRK_FEN = "8/8/8/4k3/8/8/8/4K2R w - - 0 1"
CHECKMATE_FEN = "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"

# Tier 4 / Tier 6 test positions
QUEEN_GONE_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"
DOUBLED_PAWNS_FEN = "4k3/8/8/4P3/4P3/8/8/4K3 w - - 0 1"
PASSED_PAWN_FEN = "4k3/8/8/8/8/4P3/8/4K3 w - - 0 1"
ISOLATED_PAWN_FEN = "4k3/pppppppp/8/8/4P3/8/8/4K3 w - - 0 1"
