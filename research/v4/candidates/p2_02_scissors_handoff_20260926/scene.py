"""Exact P2-02 configuration on qualified P2 natural right-arm carrier."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json
from research.v4.candidates.p2_v42_t3_20260925 import scene_blade_side_v3 as base
from research.v4.candidates.release_ketchup_region_assist_20260926.region_assist import AuthoredRegionGraspAssist,AxisSegment
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTRUCTION='Hand me the scissors.'
ASSET=ROOT/'research/v4/candidates/p1_03_scissors_case_20260926/assets/scissors.xml'
# Actual upper polymer bridge surface, radius6mm. No floating +28mm point.
BINDING=replace(base.KNIFE_BINDING,xml=str(ASSET.relative_to(ROOT)),grasp_local=(-.034,.008,.005))
REGIONS=(AxisSegment((-.043,.011,.005),(-.025,.005,.005)),)
class P2ScissorsFacade(base.P2Facade):
 instance_id='P2-02';instruction=INSTRUCTION;grasp_protocol='v4.2-assisted-authored-region-scissors-polymer-bridge-v1'
 def __init__(self,*args,**kwargs):
  r=next(x for x in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if x['id']==self.instance_id)
  if r['instructions']!=[INSTRUCTION]:raise ValueError('registry drift')
  with patch.object(base,'KNIFE_BINDING',BINDING):super().__init__(*args,**kwargs)
  obj=self.backend.objects_dict[self.item_name]
  self.grasp_assist=AuthoredRegionGraspAssist(self.backend.sim.model._model,self.backend.sim.data._data,authored_regions={obj.root_body:REGIONS},**self.backend.grasp_assist_kwargs)
