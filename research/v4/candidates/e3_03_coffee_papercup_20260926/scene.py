"""Exact E3-03 configuration; same controller, whole human and event."""
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.accelerated.e3_facade import E3TabletopFacade
from research.v4.candidates.release_ketchup_region_assist_20260926.region_assist import AuthoredRegionGraspAssist,AxisSegment
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];INSTRUCTION='Pour the coffee into the paper cup.'
BINDINGS=tuple(replace(b,xml=str((HERE/('coffee_carafe.xml'if b.name=='hot_water_kettle_1'else'paper_cup.xml')).relative_to(ROOT)),**({'grasp_local':(-.088,0.,.081)}if b.name=='hot_water_kettle_1'else{}))for b in tabletop.PILOTS['E3-01'])
REGIONS=(AxisSegment((-.088,0.,.050),(-.088,0.,.112)),)
class E3CoffeeFacade(E3TabletopFacade):
 task_id='E3-03';instruction=INSTRUCTION;grasp_protocol='v4.2-assisted-authored-region-coffee-carafe-handle-v1'
 def __init__(self,seed=0,**kwargs):
  r=next(r for r in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if r['id']=='E3-03')
  if r['instructions']!=[INSTRUCTION]:raise ValueError('registry drift')
  with patch.dict(tabletop.PILOTS,{'E3-01':BINDINGS}),patch.object(type(self),'task_id','E3-01'):super().__init__(seed,**kwargs)
  obj=self.backend.objects_dict[self.item_name];self.grasp_assist=AuthoredRegionGraspAssist(self.backend.sim.model._model,self.backend.sim.data._data,authored_regions={obj.root_body:REGIONS},**self.backend.grasp_assist_kwargs)
