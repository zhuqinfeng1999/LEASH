"""Own original foil + original normal front microwave; one private side zone."""
import copy,math
from unittest.mock import patch
import numpy as np
from research.v4.candidates.r3_04_foil_sandwich_20260926 import scene as own
from research.v4.candidates.r3_v42_tabletop_20260925 import scene as carrier
from research.v4.candidates.r3_05_oven_only_prefix_20260928.scene import OwnCartEnv
ORIGINAL_CONFIGURATION=own.configuration
SIDE_POINT=(-.036459287107,0.,.018)
OVEN_TRANSLATION=(.37875321006107665,-.17,0.)
def configuration():
 c=copy.deepcopy(ORIGINAL_CONFIGURATION())
 if c['id']!='R3-04' or c['metal_item']['grasp_local']!=[0,0,.018]:raise ValueError('Own region source drift')
 if c['microwave']['xml']!='research/v4/candidates/r3_v42_tabletop_20260925/microwave_fullsize_180.xml':raise ValueError('Original normal microwave source drift')
 c['metal_item']['grasp_local']=list(SIDE_POINT)
 return c
class HorizontalSideFoilScene(own.T1R3FoilSandwichScene):
 layout_version='R304-original-normal-front-slot-original-source-one-side-region-v1'
 def __init__(self,*args,**kwargs):
  with patch.object(own,'configuration',configuration),patch.object(carrier,'_R3FixtureEnv',OwnCartEnv):super().__init__(*args,**kwargs)
  sim=self.backend.sim;obj=self.backend.objects_dict[self.oven_name];q=np.asarray(sim.data.get_joint_qpos(obj.joints[0]),float).copy();q[:3]+=OVEN_TRANSLATION
  w,x,y,z=q[3:];c=math.cos(-math.pi/4);s=math.sin(-math.pi/4);q[3:]=[c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w]
  sim.data.set_joint_qpos(obj.joints[0],q);self._anchor_and_open_microwave()
  if kwargs.get('camera_observations',False):self.raw=self.backend._get_observations(force_update=True)
