"""Own accepted rim/PARK/physical observer stack, oven-only opt-in."""
import math
import numpy as np
from unittest.mock import patch
from research.v4.candidates.r3_06_side55_entry_20260926.scripts import RecordedScene as OriginalRecordedScene
from research.v4.candidates.r3_v42_tabletop_20260925 import scene as carrier
from research.v4.candidates.r3_05_oven_only_prefix_20260928.scene import OwnCartEnv
from .layout import TRANSLATION
class OvenOnlyTrayScene(OriginalRecordedScene):
 layout_version='R306-own-rim-PARK-real-side55-prefix-device-front-cart-v1'
 def __init__(self,*a,**kw):
  with patch.object(carrier,'_R3FixtureEnv',OwnCartEnv):super().__init__(*a,**kw)
  sim=self.backend.sim;obj=self.backend.objects_dict[self.oven_name];q=np.asarray(sim.data.get_joint_qpos(obj.joints[0]),float).copy();q[:3]+=TRANSLATION
  w,x,y,z=q[3:];c=math.cos(-math.pi/4);s=math.sin(-math.pi/4);q[3:]=[c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w]
  sim.data.set_joint_qpos(obj.joints[0],q);self._anchor_and_open_microwave()
  if kw.get('camera_observations',False):self.raw=self.backend._get_observations(force_update=True)
