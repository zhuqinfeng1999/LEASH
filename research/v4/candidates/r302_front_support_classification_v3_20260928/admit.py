import hashlib
import json
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parents[3]


def verify():
    receipt = json.loads((H / 'source_freeze_before.json').read_text())
    drift = [p for p, expected in receipt['source_sha256'].items()
             if not (ROOT / p).is_file()
             or hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != expected]
    if drift:
        raise RuntimeError('Exact classification-v3 source admission failed: ' + repr(drift))
    return receipt
