"""One visible fixed wood fixture, unchanged pillow/crib and public carrier."""
import json, math, xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.accelerated import tabletop
from research.v4.candidates.r4_04_analytical_supported_layout_20260927.scene import Scene as ParentScene

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
POINT=np.array([-.296,0.,.055])
HEIGHT=.060
NOMINAL=np.array([.080,-.183955466412554])
ORIGINAL_FACTORY=tabletop._make_env_class

def make_env():
 base=ORIGINAL_FACTORY()
 class WoodEnv(base):
  def _load_model(self):
   super()._load_model()
   xyz,yaw=tabletop.sample_slot('L',self.layout_seed,object_index=0)
   xyz[:2]+=NOMINAL-np.array([-.08,-.24]);xyz[2]=.9
   root=ET.parse(HERE/'assets/wood_riser.xml').getroot()
   for x in root.find('asset'):self.model.asset.append(x)
   body=root.find('worldbody/body');body.set('pos',' '.join(map(str,xyz)));body.set('quat',f'{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}')
   self.model.worldbody.append(body)
 return WoodEnv

class Scene(ParentScene):
 protocol='R404_VISIBLE_REAL60MM_WOOD45X35_BEVEL_HORIZONTAL_SEAM_COPY_ONLY_V1'
 def __init__(self,*a,**kw):
  with patch.object(tabletop,'_make_env_class',make_env):super().__init__(*a,**kw)
  try:
   sim=self.backend.sim;obj=self.backend.objects_dict[self.item_name]
   q=sim.data.get_joint_qpos(obj.joints[0]).copy();q[0]+=.080;q[2]+=HEIGHT
   sim.data.set_joint_qpos(obj.joints[0],q);sim.data.set_joint_qvel(obj.joints[0],np.zeros(6))
   sim.model.site_pos[sim.model.site_name2id('v42_adult_pillow_1_grasp_zone')]=POINT
   sim.forward();self.initial_item_z=float(q[2]);self.candidate_wood_initial=dict(pillow_qpos=q.tolist(),wood_nominal_xy_m=NOMINAL.tolist(),wood_height_m=HEIGHT,original_region_body_m=POINT.tolist(),actual_controls=0,wood_static_separate_body=True,pillow_weld_not_activated=True)
   if self.backend.use_camera_obs:self.raw=self.backend._get_observations(force_update=True)
  except BaseException:super().close();raise
