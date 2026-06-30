"""Legacy SFT data-generation package bootstrap.

The modules under this directory historically used script-directory imports
such as ``config.settings`` and ``generators.base``.  Keep those imports
working when the same files are imported as ``sft.make_data.*`` from the repo
root during the migration to ``chess_llm``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_MAKE_DATA_ROOT = Path(__file__).resolve().parent
if str(_MAKE_DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAKE_DATA_ROOT))

__all__: list[str] = []
