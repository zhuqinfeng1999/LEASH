import hashlib,json
from pathlib import Path
from research.v4.candidates.p1_natural_right_v7_20260925 import run_method as live
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def hashes():
    paths={ROOT/name for name in live.SOURCE_FILES}
    for directory in live.ASSET_DIRS:paths.update(p for p in(ROOT/directory).rglob('*')if p.is_file())
    for directory in(HERE,ROOT/'research/v4/candidates/p1_visible_blade_20260926'):
        paths.update(p for p in directory.glob('*.py'))
    paths.update(p for p in(HERE/'assets').glob('*')if p.is_file());paths.add(HERE/'protocol.json')
    result={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(paths)}
    freeze=HERE/'source_freeze.json'
    if freeze.exists()and result!=json.loads(freeze.read_text()):raise ValueError('P1-05 source drift')
    return result
