"""Normal static closed scissors; genuine adjacent loop holes and layered blades."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json
from research.v4.candidates.p2_v42_t3_20260925 import scene_safe_slot_v2 as base
from research.v4.candidates.release_ketchup_region_assist_20260926.region_assist import AuthoredRegionGraspAssist,AxisSegment
from research.v4.candidates.p2_03_visible_screwdriver_20260926.scene import expose
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTRUCTION='Hand me the scissors.'
ASSET=HERE/'assets/closed_scissors.xml'
BINDING=replace(base.KNIFE_BINDING,xml=str(ASSET.relative_to(ROOT)),z_offset=.007,grasp_local=(-.034,.008,.005))
REGIONS=(AxisSegment((-.043,.013,.005),(-.025,.005,.005)),)
class P2ClosedScissorsFacade(base.P2Facade):
 instance_id='P2-02';instruction=INSTRUCTION;grasp_protocol='v4.2-assisted-authored-region-closed-scissors-polymer-bridge-v1'
 def __init__(self,*args,**kwargs):
  row=next(x for x in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if x['id']==self.instance_id)
  if row['instructions']!=[INSTRUCTION]:raise ValueError('registry drift')
  with patch.object(base,'KNIFE_BINDING',BINDING):super().__init__(*args,**kwargs)
  obj=self.backend.objects_dict[self.item_name]
  self.grasp_assist=AuthoredRegionGraspAssist(self.backend.sim.model._model,self.backend.sim.data._data,authored_regions={obj.root_body:REGIONS},**self.backend.grasp_assist_kwargs)
  expose(self)
