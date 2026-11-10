"""Exact registered English text and accepted own-source provenance."""
from dataclasses import dataclass
from importlib import import_module
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
IDS=tuple(f'S1-0{i}'for i in range(1,6))
@dataclass(frozen=True)
class Binding:
    instance:str
    noun:str
    module:str
    class_name:str
    acceptance:str
ROWS=(
 Binding('S1-01','pan','scene_review_s1_01_authored_20260927.future_scene','S1AuthoredReleaseFacade','s1_01_authored_pan_native_root_review_20260927.json'),
 Binding('S1-02','saucepan','s1_02_saucepan_t8_20260926.scene','S1SaucepanFacade','s1_02_v42_instance_acceptance_20260926.json'),
 Binding('S1-03','kettle','s1_03_kettle_t8_20260926.scene','S1KettleFacade','s1_03_v42_instance_acceptance_20260926.json'),
 Binding('S1-04','ladle','s1_04_ladle_t8_20260926.scene','S1LadleFacade','s1_04_v42_instance_acceptance_20260926.json'),
 Binding('S1-05','griddle','s1_05_griddle_t8_20260926.scene','S1GriddleFacade','s1_05_v42_instance_acceptance_20260926.json'))
def binding(instance):
    b=next((b for b in ROWS if b.instance==instance),None)
    if b is None:raise ValueError('unregistered S1 instance')
    row=next(r for r in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if r['id']==instance)
    expected=(f'Move the {b.noun} onto the wooden board.',f'Put the {b.noun} in my '+('hand.'if instance=='S1-04'else'hands.'))
    if tuple(row['instructions'])!=expected:raise ValueError('registered instruction drift')
    module=import_module('research.v4.candidates.'+b.module)
    return b,expected,getattr(module,b.class_name)
def accepted_sources(instance):
    b,_,_=binding(instance);path=ROOT/'research/v4/results'/b.acceptance
    report=json.loads(path.read_text())
    if instance=='S1-01':
        if not report['pan_native_appearance_pass']or report['source_drift']:raise ValueError('authored pan not accepted')
        # Root accepted this exact authored facade for future S101; the HD map
        # includes the accepted physical/complete-visual providers and 4K material.
        pins=json.loads((ROOT/'research/v4/candidates/scene_review_s1_01_authored_20260927/source_freeze.json').read_text())
    else:
        if not report['accepted']or report['source_drift']:raise ValueError('own engineering qualification missing')
        pins=dict(report['source_hashes'])
    for name,sha in pins.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('accepted source drift: '+name)
    pins[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pins
