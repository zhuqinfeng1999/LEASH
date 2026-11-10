import hashlib,inspect,json
from pathlib import Path
from . import binding
from research.v4.candidates.release_appearance_composed_factory_20260927 import freeze as parent
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes():
    pins=parent.hashes();paths=[*HERE.glob('*.py'),HERE/'HANDOFF.md',HERE/'cpu_summary.json',HERE/'binding_contract.json',parent.HERE/'source_freeze.json',
        ROOT/'research/v4/candidates/release_r2_remaining_staged_readonly_20260927/source_freeze.json']
    # Existing base method function supplies own shared runtime/source files.
    # These current hashes are a new appearance receipt, not inherited scene
    # or live-method qualification from old historical source hashes.
    for name,h in binding.r202.hashes().items():
        if name in pins and pins[name]!=h:raise RuntimeError('Qualified parent/current input conflict '+name)
        pins[name]=h
    for i in binding.INSTANCES:
        c=binding.own(i);paths.extend([HERE/i/'cpu_contract.json',Path(c.entry.__file__)])
        for cls in (c.Evaluator,*c.Scene.__mro__):
            try:p=Path(inspect.getfile(cls)).resolve()
            except TypeError:continue
            if p.is_relative_to(ROOT)and not p.is_relative_to(ROOT/'envs'):paths.append(p)
        carrier=ROOT/'research/v4/candidates'/f'{i.lower().replace("-","_")}_t1_derived_20260926'
        paths.extend(carrier.glob('*.py'));paths.extend(p for p in carrier.glob('*.json')if p.name.startswith(('config','selected_pose')))
        paths.extend(p for p in (carrier/'assets').glob('*')if p.is_file())
        paths.append(ROOT/f'research/v4/results/{i.lower().replace("-","_")}_v42_instance_acceptance_20260926.json')
    for mod in (binding.r203,binding.remaining):paths.extend(Path(mod.__file__).parent.glob('*.py'))
    for p in paths:pins[str(p.relative_to(ROOT))]=digest(p)
    pins=dict(sorted(pins.items()))
    for n,h in pins.items():
        if digest(ROOT/n)!=h:raise RuntimeError('Source drift '+n)
    old=HERE/'source_freeze.json'
    if old.exists()and json.loads(old.read_text())['source_sha256']!=pins:raise RuntimeError('Own R2 candidate drift')
    return pins
if __name__=='__main__':
    pins=hashes();r=dict(source_sha256=pins,source_count=len(pins),canonical_sha256=hashlib.sha256(json.dumps(pins,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        own_instances=list(binding.INSTANCES),default_adopted=False,native_pending=True)
    (HERE/'source_freeze.json').write_text(json.dumps(r,indent=2)+'\n');print(r['source_count'],r['canonical_sha256'])
