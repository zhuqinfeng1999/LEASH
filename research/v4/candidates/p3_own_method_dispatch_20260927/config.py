"""Exact registered P3 nouns and accepted owner-side factories; no coordinates."""
import importlib,json,hashlib
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
@dataclass(frozen=True)
class Binding:
    instance:str;module:str;source:str;source_prompt:str;source_aliases:tuple
    protected:str;protected_prompt:str;protected_query2:str;target:str
    acceptance:str

ROWS=(
 Binding('P3-01','research.v4.accelerated.p3_facade','cup','cup',('cup','water'),
  'laptop','laptop computer','computer','coaster','p3_01_current_source_v42_instance_acceptance_20260927.json'),
 Binding('P3-02','research.v4.candidates.p3_02_keyboard_coffee_20260926.scene','coffee','coffee cup',('coffee','cup','coffee cup'),
  'keyboard','mechanical keyboard','keyboard','coaster','p3_02_v42_instance_acceptance_20260926.json'),
 Binding('P3-03','research.v4.candidates.p3_03_tablet_juice_20260926.scene','juice','juice cup',('juice','cup','juice cup'),
  'tablet','tablet computer','tablet','coaster','p3_03_v42_instance_acceptance_20260926.json'),
 Binding('P3-04','research.v4.candidates.p3_04_album_wine_v2_20260926.scene','wine','wine glass',('wine','glass','wine glass'),
  'photo album','open photo album','photo album','coaster','p3_04_v42_instance_acceptance_20260926.json'),
 Binding('P3-05','research.v4.candidates.p3_05_book_soup_20260926.scene','soup','soup bowl',('soup','bowl','soup bowl'),
  'book','open book','book','placemat','p3_05_v42_instance_acceptance_20260926.json'))
IDS=tuple(x.instance for x in ROWS)
def binding(instance):
    b=next((x for x in ROWS if x.instance==instance),None)
    if b is None:raise ValueError('unregistered P3 instance')
    reg=json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']
    row=next(x for x in reg if x['id']==instance)
    if row['base_task']!='P3' or len(row['instructions'])!=1:raise ValueError('registry mismatch')
    m=importlib.import_module(b.module)
    expected=getattr(m,'INSTRUCTION',row['instructions'][0])
    if expected!=row['instructions'][0]:raise ValueError('exact English instruction mismatch')
    return b,row['instructions'][0],getattr(m,'Scene',getattr(m,'P3TabletopFacade',None))
def accepted_sources(instance):
    b,_,_=binding(instance);p=ROOT/'research/v4/results'/b.acceptance
    a=json.loads(p.read_text())
    if a['accepted'] is not True or a['instance']!=instance:raise ValueError('not currently accepted')
    pins=dict(a['source_hashes'])
    for name,sha in pins.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('accepted own source drift: '+name)
    for q in (p,ROOT/a['scripts']):pins[str(q.relative_to(ROOT))]=hashlib.sha256(q.read_bytes()).hexdigest()
    return pins
