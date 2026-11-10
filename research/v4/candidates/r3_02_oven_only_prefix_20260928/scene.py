"""Opt-in actual R302 original carrier/source/PARK; only full appliance/cart."""
import math
import numpy as np
from unittest.mock import patch
from research.v4.candidates.r3_02_cc0_soup_config_20260926.scene import SoupCanScene
from research.v4.candidates.r3_v42_tabletop_20260925 import scene as carrier
from research.v4.candidates.r3_05_oven_only_prefix_20260928.scene import OwnCartEnv
from .layout import TRANSLATION,LAYOUT

class OvenOnlySoupScene(SoupCanScene):
 layout_version='R302-own-source-can-PARK-original-prefix-device-front-cart-v1'
 def __init__(self,*args,**kwargs):
  with patch.object(carrier,'_R3FixtureEnv',OwnCartEnv):super().__init__(*args,**kwargs)
  sim=self.backend.sim;obj=self.backend.objects_dict[self.oven_name]
  q=np.asarray(sim.data.get_joint_qpos(obj.joints[0]),float).copy()
  self.oven_only_original_qpos=q.copy()
  q[:3]+=np.asarray(TRANSLATION)
  w,x,y,z=q[3:];c=math.cos(-math.pi/4);s=math.sin(-math.pi/4)
  q[3:]=[c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w]
  sim.data.set_joint_qpos(obj.joints[0],q)
  self._anchor_and_open_microwave() # Existing original anchor/reset; no step.
  self.oven_only_selected_qpos=np.asarray(sim.data.get_joint_qpos(obj.joints[0]),float).copy()
  if not np.array_equal(self.oven_only_selected_qpos,q):raise ValueError('Appliance transform/anchor not retained')
  if kwargs.get('camera_observations',False):self.raw=self.backend._get_observations(force_update=True)
