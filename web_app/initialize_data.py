"""Thin wrapper — canonical script: scripts/initialize_web_data.py"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "initialize_web_data.py"

if __name__ == "__main__":
    sys.argv[0] = str(_SCRIPT)
    runpy.run_path(str(_SCRIPT), run_name="__main__")
