"""Complete qualified622 plus own real main seam, exact manifest admission."""
import hashlib,json
from pathlib import Path
from research.v4.candidates.release_r2_remaining_method_appearance_20260927 import freeze as appearance
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
FILES=('binding.py','freeze.py','check_cpu.py','inspect_entry.py','HANDOFF.md','cpu_contract.json')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def expected_current():
    pins=appearance.hashes()
    pins[str((appearance.HERE/'source_freeze.json').relative_to(ROOT))]=digest(appearance.HERE/'source_freeze.json')
    for name in FILES:pins[str((HERE/name).relative_to(ROOT))]=digest(HERE/name)
    return dict(sorted(pins.items()))
def manifest():return json.loads((HERE/'source_freeze.json').read_text())
def hashes():
    current=expected_current();m=manifest()
    if m.get('source_sha256')!=current or m.get('source_count')!=len(current):raise RuntimeError('Own real-method source admission mismatch')
    canonical=hashlib.sha256(json.dumps(current,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if m.get('canonical_sha256')!=canonical:raise RuntimeError('Own manifest canonical hash mismatch')
    return current
def admit_sources():
    # Original real main consumes this return value before services/Scene.
    # Also pin this manifest itself in episode source changes-after checks,
    # without circularly including its bytes in its own manifest.
    pins=hashes();pins[str((HERE/'source_freeze.json').relative_to(ROOT))]=digest(HERE/'source_freeze.json')
    return dict(sorted(pins.items()))
def build():
    pins=expected_current();r=dict(source_sha256=pins,source_count=len(pins),canonical_sha256=hashlib.sha256(json.dumps(pins,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        real_main_executed=False,Scene_constructors=0,models_services=0,adopted=False,instance='R2-02')
    if(HERE/'source_freeze.json').exists():raise RuntimeError('Do not overwrite a frozen candidate')
    (HERE/'source_freeze.json').write_text(json.dumps(r,indent=2)+'\n');print(r['source_count'],r['canonical_sha256'])
if __name__=='__main__':build()
