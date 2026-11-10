"""P1-06 normal assets over frozen naturalwhole-right-arm T2carrier."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json
from research.v4.accelerated import tabletop
from research.v4.candidates.p1_table_wrist_clearance_20260925.candidate import P1TableWristClearanceFacade
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTRUCTION='Put the wine bottle in the ice bucket on the right.'
BINDINGS=tuple(replace(b,xml=str((HERE/'assets'/('wine_bottle.xml'if b.name=='blunt_tool_1'else'ice_bucket.xml')).relative_to(ROOT)),grasp_local=(0,0,.240)if b.name=='blunt_tool_1'else b.grasp_local,z_offset=0 if b.name=='blunt_tool_1'else b.z_offset)for b in tabletop.PILOTS['E1-01'])
class P1WineBucketFacade(P1TableWristClearanceFacade):
 instance_id='P1-06';instruction=INSTRUCTION
 def __init__(self,*args,**kwargs):
  r=next(x for x in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if x['id']=='P1-06')
  if r['slots']!={'item':'玻璃酒瓶','target':'冰桶'}or r['instructions']!=[INSTRUCTION]:raise ValueError('exactregistrychanged')
  with patch.dict(tabletop.PILOTS,{'E1-01':BINDINGS}):super().__init__(*args,**kwargs)
