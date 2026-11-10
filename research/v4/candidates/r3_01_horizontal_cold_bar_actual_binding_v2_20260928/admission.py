"""Complete source admission; frozen before Root's first constructor."""
from research.v4.leash_v43.run_manifest import source_custody_enabled
import hashlib
import json
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admit():
    if not source_custody_enabled():
        return {}
    pins = json.loads((H / 'source_freeze_final.json').read_text())
    drift = []
    for name, digest in pins.items():
        p = Path(name)
        p = p if p.is_absolute() else ROOT / p
        if not p.is_file() or sha(p) != digest:
            drift.append(name)
    if drift:
        raise ValueError('Own R301 source drift: ' + ', '.join(drift[:8]))
    return {'source_entries': len(pins), 'source_drift': [],
            'actual_manifest_file_sha256': sha(H / 'source_freeze_final.json')}
