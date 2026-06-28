"""Root conftest — ensures sft/training/ is on sys.path for all tests."""

import sys
from pathlib import Path

_TRAINING_ROOT = str(Path(__file__).resolve().parent)
if _TRAINING_ROOT not in sys.path:
    sys.path.insert(0, _TRAINING_ROOT)
