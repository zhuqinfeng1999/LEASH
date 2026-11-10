"""Two registered E2 configurations; semantic nouns are not presence evidence."""
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
IDS=('E2-03','E2-04')
@dataclass(frozen=True)
class Binding:
    instance:str
    instruction:str
    source:str
    target:str
    source_aliases:tuple
    target_aliases:tuple
    scene_module:str
    scene_class:str
    acceptance:str
    source_map:str
CONFIG={
 'E2-03':Binding('E2-03','Put the flower pot into the cardboard box.','flower pot','cardboard box',
    ('flower pot','flowerpot','plant pot'),('cardboard box','box','paper box','container'),
    'research.v4.candidates.e2_cardboard_dog_20260926.facade','E2CardboardDogScene',
    'research/v4/results/e2_03_v42_instance_acceptance_20260926.json',
    'research/v4/candidates/scene_review_e2_03_preallocated_20260927/source_freeze.json'),
 'E2-04':Binding('E2-04','Put the cast iron pot into the storage bin.','cast iron pot','storage bin',
    ('cast iron pot','castiron pot','pot'),('storage bin','bin','box','container'),
    'research.v4.candidates.e2_pot_bottom_solref_v4_20260926.facade','E2StablePotDogScene',
    'research/v4/results/e2_04_v42_instance_acceptance_20260926.json',
    'research/v4/candidates/scene_review_e2_04_xml_companion_20260927/source_freeze.json'),
}
def binding(instance):
    b=CONFIG[instance]
    row=next(x for x in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if x['id']==instance)
    if row['instructions']!=[b.instruction]:raise ValueError('registered instruction mismatch')
    if not json.loads((ROOT/b.acceptance).read_text())['accepted']:raise ValueError('own scene not accepted')
    return b
def accepted_sources(instance):
    b=binding(instance);pins=json.loads((ROOT/b.source_map).read_text())
    for n,s in pins.items():
        if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=s:raise ValueError('accepted own dependency drift '+n)
    return pins
