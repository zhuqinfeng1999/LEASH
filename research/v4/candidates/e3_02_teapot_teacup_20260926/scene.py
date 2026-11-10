"""Exact E3-02 thin configuration; original live full-person event unchanged."""
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.accelerated.e3_facade import E3TabletopFacade
from research.v4.candidates.release_ketchup_region_assist_20260926.region_assist import AuthoredRegionGraspAssist,AxisSegment
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTRUCTION='Pour the tea into the teacup.'
BINDINGS=tuple(replace(b,xml=str((HERE/('teapot.xml'if b.name=='hot_water_kettle_1'else'teacup.xml')).relative_to(ROOT)))for b in tabletop.PILOTS['E3-01'])
REGIONS=(AxisSegment((-.104,0.,.046),(-.104,0.,.100)),)
class E3TeapotFacade(E3TabletopFacade):
 task_id='E3-02';instruction=INSTRUCTION;grasp_protocol='v4.2-assisted-authored-region-teapot-handle-v1'
 def __init__(self,seed=0,**kwargs):
  row=next(r for r in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if r['id']=='E3-02')
  if row['instructions']!=[INSTRUCTION]:raise ValueError('exact registry drift')
  with patch.dict(tabletop.PILOTS,{'E3-01':BINDINGS}),patch.object(type(self),'task_id','E3-01'):super().__init__(seed,**kwargs)
  obj=self.backend.objects_dict[self.item_name];self.grasp_assist=AuthoredRegionGraspAssist(self.backend.sim.model._model,self.backend.sim.data._data,authored_regions={obj.root_body:REGIONS},**self.backend.grasp_assist_kwargs)
