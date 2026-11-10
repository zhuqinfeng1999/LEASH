"""Thin T2 P3 registry binding; private compatibility body does not name policy."""
import copy,json
from pathlib import Path
from dataclasses import replace
import xml.etree.ElementTree as E
from unittest.mock import patch
from research.v4.accelerated import tabletop,p3_facade
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTRUCTION='Put the soup on the placemat on the right.'

def book(arena):
    root=E.parse(HERE/'book.xml').getroot()
    for x in root.find('asset'):arena.asset.append(copy.deepcopy(x))
    arena.worldbody.append(copy.deepcopy(root.find('worldbody/body')))

_ORIGINAL_CREATE=p3_facade._create_p3_scene
# Bind the original to avoid recursion while the parent factory is patched.
def create_scene(**kwargs):
    bindings=tuple(replace(b,xml=str((HERE/'bowl.xml').relative_to(ROOT)),grasp_local=(-.065,0.,.048))if b.name=='liquid_proxy_cup_1'else replace(b,xml=str((HERE/'placemat.xml').relative_to(ROOT)))if b.name=='p3_coaster_1'else b for b in tabletop.PILOTS['P3-01'])
    with patch.object(tabletop,'PILOTS',{**tabletop.PILOTS,'P3-01':bindings}),patch.object(tabletop,'_append_p3_laptop',book):return _ORIGINAL_CREATE(**kwargs)
class Scene(p3_facade.P3TabletopFacade):
    def __init__(self,*args,**kwargs):
        row=next(x for x in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if x['id']=='P3-05')
        if row['instructions']!=[INSTRUCTION]:raise ValueError('registry mismatch')
        with patch.object(p3_facade,'_create_p3_scene',create_scene):super().__init__(*args,**kwargs)
