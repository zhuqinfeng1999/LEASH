"""English public instructions for the S3 multi-round method entry."""
from __future__ import annotations

import json
from pathlib import Path

INSTRUCTIONS = tuple(json.loads((Path(__file__).parent / "task.json").read_text())["rounds"])
if len(INSTRUCTIONS) != 3 or any(not isinstance(x, str) or not x for x in INSTRUCTIONS):
    raise ValueError("S3 requires exactly three English instructions")
