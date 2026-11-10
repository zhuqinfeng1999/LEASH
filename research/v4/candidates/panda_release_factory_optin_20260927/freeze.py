"""Source union of approved142 and actually exercised current factories."""
from pathlib import Path
import hashlib,json
from research.v4.candidates.panda_front_edge_flip_20260927 import proof as accepted
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def hashes():
    pins=dict(accepted.hashes())
    current_S1=json.loads((ROOT/'research/v4/candidates/scene_review_s1_01_authored_20260927/source_freeze.json').read_text())
    # Root's current accepted E102 factory dependencies; preserve historical
    # hashes in that original record, record current bytes as proposal inputs.
    e102_path=ROOT/'research/v4/results/e1_02_v42_instance_acceptance_20260926.json'
    e102=json.loads(e102_path.read_text());current_E1={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()for n in e102['source_hashes']}
    for mapping in (current_S1,current_E1):
        for n,h in mapping.items():
            if n in pins and pins[n]!=h:raise RuntimeError('Source union conflict '+n)
            pins[n]=h
    paths=[*HERE.glob('*.py'),HERE/'cpu_contract.json',HERE/'scope_contract.json',HERE/'NOTICE.md',HERE/'HANDOFF.md',
           e102_path,ROOT/'research/v4/candidates/scene_review_s1_01_authored_20260927/source_freeze.json',accepted.HERE/'source_freeze.json']
    for p in paths:pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for n,h in pins.items():
        p=ROOT/n if not Path(n).is_absolute()else Path(n)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise RuntimeError('Current source drift '+n)
    return dict(sorted(pins.items()))
if __name__=='__main__':
    pins=hashes();packet={'source_sha256':pins,'source_count':len(pins),'canonical_sha256':hashlib.sha256(json.dumps(pins,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'appearance_version':'panda-original-MIT-plain-panel-142-optin-v1','not_applied_to_common_or_historical':True}
    (HERE/'source_freeze.json').write_text(json.dumps(packet,indent=2)+'\n');print(packet['source_count'],packet['canonical_sha256'])
