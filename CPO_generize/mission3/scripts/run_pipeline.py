from __future__ import annotations

import sys
from pathlib import Path


MISSION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MISSION_DIR / "src"))

from mission3.cli import main  # noqa: E402


raise SystemExit(main())

