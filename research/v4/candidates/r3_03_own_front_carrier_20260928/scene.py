"""Own R303 front reset/cart only; no other object's source or grasp inherited."""
import math
from unittest.mock import patch
import numpy as np
from research.v4.candidates.r3_03_t1_derived_20260926 import scene as own
from research.v4.candidates.r3_v42_tabletop_20260925 import scene as carrier
from research.v4.candidates.r3_05_oven_only_prefix_20260928.scene import OwnCartEnv

OVEN_NOMINAL=(.25029602123971006,0.,.9)
OVEN_TRANSLATION=(.29029602123971004,-.17,0.)
FRONT_YAW=-math.pi/2

class OwnFrontThermosScene(own.T1R3ThermosScene):
    layout_version='R303-original-thermos-source-PARK-normal-front-oven-existing-cart-v1'
    def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
        # The own constructor retains its original source, NEUTRAL reset,
        # +59mm site, bindings, assets and GraspAssist construction order.
        with patch.object(carrier,'_R3FixtureEnv',OwnCartEnv):
            super().__init__(seed,camera_observations=camera_observations,
                             render_gpu_device_id=render_gpu_device_id)
        sim=self.backend.sim
        obj=self.backend.objects_dict[self.oven_name]
        q=np.asarray(sim.data.get_joint_qpos(obj.joints[0]),float).copy()
        q[:3]+=OVEN_TRANSLATION
        w,x,y,z=q[3:];c=math.cos(FRONT_YAW/2);s=math.sin(FRONT_YAW/2)
        q[3:]=[c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w]
        sim.data.set_joint_qpos(obj.joints[0],q)
        self._anchor_and_open_microwave() # Existing reset-only fixture seam.
        if camera_observations:self.raw=self.backend._get_observations(force_update=True)
