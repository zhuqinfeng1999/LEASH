"""Pure opt-in binding-data replacement, no constructor/default mutation."""
from dataclasses import asdict,replace
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
ORIGINAL=ROOT/'research/v4/candidates/r3_v42_tabletop_20260925/microwave_fullsize_180.xml'
CANDIDATE=str((H/'microwave_tall60.xml').relative_to(ROOT))
def select(bindings):
 targets=[x for x in bindings if x.name=='microwave_1']
 if len(targets)!=1:raise ValueError('One explicit current microwave binding required')
 old=targets[0];p=Path(old.xml);p=p if p.is_absolute()else ROOT/p
 if p.resolve()!=ORIGINAL.resolve()or not old.fixed:raise ValueError('Unknown asset/family/source-stage; fail closed')
 new=replace(old,xml=CANDIDATE);a=asdict(old);b=asdict(new);a.pop('xml');b.pop('xml')
 if a!=b:raise ValueError('Nonasset binding field changed')
 return tuple(new if x is old else x for x in bindings)
