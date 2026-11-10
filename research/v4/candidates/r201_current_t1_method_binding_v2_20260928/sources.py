"""Complete immutable source admission after the selected binding exists."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def admit():
    pins = json.loads((HERE / 'source_freeze.json').read_text())
    for name, digest in pins.items():
        p = Path(name) if name.startswith('/') else ROOT / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Current R201 v2 source/evidence drift: ' + name)
    return pins
