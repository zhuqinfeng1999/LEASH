"""Normal original ice pick on the accepted natural T3 table carrier."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json
from research.v4.candidates.p2_v42_t3_20260925 import scene_safe_slot_v2 as base
from research.v4.candidates.release_ketchup_region_assist_20260926.region_assist import AuthoredRegionGraspAssist,AxisSegment
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTRUCTION='Hand me the ice pick.'
ASSET=HERE/'assets/ice_pick.xml'
BINDING=replace(base.KNIFE_BINDING,xml=str(ASSET.relative_to(ROOT)),z_offset=.014,grasp_local=(-.064,0.,.009))
REGIONS=(AxisSegment((-.080,0.,.009),(-.040,0.,.009)),)
class P2IcePickFacade(base.P2Facade):
 instance_id='P2-04';instruction=INSTRUCTION;grasp_protocol='v4.2-assisted-authored-region-icepick-handle-v1'
 def __init__(self,*args,**kwargs):
  row=next(x for x in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if x['id']==self.instance_id)
  if row['instructions']!=[INSTRUCTION]:raise ValueError('registry drift')
  with patch.object(base,'KNIFE_BINDING',BINDING):super().__init__(*args,**kwargs)
  obj=self.backend.objects_dict[self.item_name]
  self.grasp_assist=AuthoredRegionGraspAssist(self.backend.sim.model._model,self.backend.sim.data._data,authored_regions={obj.root_body:REGIONS},**self.backend.grasp_assist_kwargs)
