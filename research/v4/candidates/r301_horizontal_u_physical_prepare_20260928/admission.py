"""Complete versioned source admission; historical manifests remain immutable."""
from research.v4.leash_v43.run_manifest import source_custody_enabled
from pathlib import Path
import hashlib
import json

H = Path(__file__).resolve().parent
ROOT = H.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_sources(sources):
    required = json.loads((H / 'parent_membership.json').read_text())['required_paths']
    if not set(required).issubset(sources) or str(ROOT / 'AGENTS.md') not in sources:
        raise ValueError('Complete inherited source membership/AGENTS required')
    for name, expected in sources.items():
        if not Path(name).is_file() or sha(name) != expected:
            raise ValueError('Horizontal-U physical source drift: ' + name)
    return {'source_entries': len(sources), 'source_drift': []}


def admit():
    if not source_custody_enabled():
        return {}
    manifest = H / 'source_freeze.json'
    value = json.loads(manifest.read_text())
    result = validate_sources(value['sources'])
    result['actual_manifest_file_sha256'] = sha(manifest)
    result['historical_AGENTS_difference_is_versioned_not_old_freeze_pass'] = True
    return result
