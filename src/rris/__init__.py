"""Restaurant Reputation Intelligence System — shared Python library."""

from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.1.0"

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

_SRC = PACKAGE_ROOT.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
