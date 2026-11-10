"""Exact qualified-input unions and current isolated runtime, no common adoption."""
from pathlib import Path
import hashlib,json
from research.v4.candidates.panda_release_factory_optin_20260927 import freeze as panda
from research.v4.candidates.garment_camera_diagnostic_v2_20260927 import native_diagnostic as garment
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes():
    pins={}
    for src in (panda.hashes(),garment.hashes()):
        for name,h in src.items():
            path=Path(name)
            if path.is_absolute():
                try:name=str(path.relative_to(ROOT))
                except ValueError:name=str(path)
            if name in pins and pins[name]!=h:raise RuntimeError('Qualified input union conflict '+name)
            pins[name]=h
    paths=[*HERE.glob('*.py'),HERE/'NOTICE.md',HERE/'HANDOFF.md',HERE/'cpu_contract.json',HERE/'binding_contract.json',HERE/'initial_inventory.json',
       HERE/'startup_probe_accessor_failure.json',HERE/'startup_probe_accessor_failure.py.txt',
       HERE/'r2_probe_accessor_failure.json',HERE/'r2_probe_accessor_failure.py.txt',
       panda.HERE/'source_freeze.json',garment.HERE/'source_freeze.json',
       ROOT/'research/v4/results/garment_plain_cotton_native_root_review_20260927.json']
    for p in paths:pins[str(p.relative_to(ROOT))]=digest(p)
    pins=dict(sorted(pins.items()))
    for name,h in pins.items():
        if digest(ROOT/name)!=h:raise RuntimeError('Current source drift '+name)
    existing=HERE/'source_freeze.json'
    if existing.exists()and json.loads(existing.read_text())['source_sha256']!=pins:raise RuntimeError('Isolated composed source drift')
    return pins
if __name__=='__main__':
    pins=hashes();r={'source_sha256':pins,'source_count':len(pins),'canonical_sha256':hashlib.sha256(json.dumps(pins,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
      'appearance_version':'release-Panda142-cotton82-authoredS101232-optin-v1','common_adoption':False,'native_pending':True}
    (HERE/'source_freeze.json').write_text(json.dumps(r,indent=2)+'\n');print(r['source_count'],r['canonical_sha256'])
