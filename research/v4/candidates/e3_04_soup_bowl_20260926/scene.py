"""Exact E3-04 thin T6 config, inherited natural full-person lift event."""
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.accelerated.e3_facade import E3TabletopFacade
from research.v4.candidates.e3_relaxed_initial_person_20260926.scene import RelaxedMixin
from research.v4.candidates.release_ketchup_region_assist_20260926.region_assist import AuthoredRegionGraspAssist,AxisSegment
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTRUCTION='Pour the soup into the bowl.'
BINDINGS=tuple(replace(b,xml=str((HERE/('soup_pot.xml'if b.name=='hot_water_kettle_1'else'empty_bowl.xml')).relative_to(ROOT)),**({'grasp_local':(-.146,0.,.078)}if b.name=='hot_water_kettle_1'else{}))for b in tabletop.PILOTS['E3-01'])
REGIONS=(AxisSegment((-.146,-.025,.078),(-.146,.025,.078)),)
class Scene(RelaxedMixin,E3TabletopFacade):
 task_id='E3-04';instruction=INSTRUCTION;grasp_protocol='v4.2-assisted-real-steel-soup-pot-handle-v1'
 def __init__(self,seed=0,**kwargs):
  row=next(r for r in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if r['id']=='E3-04')
  if row['instructions']!=[INSTRUCTION]or row['slots']!={'pourer':'汤锅','cup':'碗'}:raise ValueError('exact registry drift')
  with patch.dict(tabletop.PILOTS,{'E3-01':BINDINGS}),patch.object(type(self),'task_id','E3-01'):super().__init__(seed,**kwargs)
  obj=self.backend.objects_dict[self.item_name];self.grasp_assist=AuthoredRegionGraspAssist(self.backend.sim.model._model,self.backend.sim.data._data,authored_regions={obj.root_body:REGIONS},**self.backend.grasp_assist_kwargs)
