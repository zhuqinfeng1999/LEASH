import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent
ROOT=H.parents[3]
def verify():
    x=json.loads((H/'source_freeze_before.json').read_text())
    drift=[p for p,v in x['source_sha256'].items() if not(ROOT/p).is_file() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=v]
    if drift:raise RuntimeError('Exact source support-pair guard-v2 admission failed: '+repr(drift))
    return x
