"""Actual existing303.5mm CC0 carving fork on natural T3 carrier."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json
from research.v4.candidates.p2_v42_t3_20260925 import scene_safe_slot_v2 as base
from research.v4.candidates.release_ketchup_region_assist_20260926.region_assist import AuthoredRegionGraspAssist,AxisSegment
from research.v4.candidates.p2_03_visible_screwdriver_20260926.scene import expose
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTRUCTION='Hand me the carving fork.'
ASSET=ROOT/'research/v4/candidates/e1_04_carving_fork_platter_20260926/assets/carving_fork.xml'
BINDING=replace(base.KNIFE_BINDING,xml=str(ASSET.relative_to(ROOT)),z_offset=.015,grasp_local=(-.073,0.,.010))
REGIONS=(AxisSegment((-.103,0.,.010),(-.043,0.,.010)),)
class P2CarvingForkFacade(base.P2Facade):
 instance_id='P2-05';instruction=INSTRUCTION;grasp_protocol='v4.2-assisted-authored-region-carving-fork-handle-v1'
 def __init__(self,*args,**kwargs):
  row=next(x for x in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if x['id']==self.instance_id)
  if row['instructions']!=[INSTRUCTION]:raise ValueError('registry drift')
  with patch.object(base,'KNIFE_BINDING',BINDING):super().__init__(*args,**kwargs)
  obj=self.backend.objects_dict[self.item_name]
  self.grasp_assist=AuthoredRegionGraspAssist(self.backend.sim.model._model,self.backend.sim.data._data,authored_regions={obj.root_body:REGIONS},**self.backend.grasp_assist_kwargs)
  expose(self)
