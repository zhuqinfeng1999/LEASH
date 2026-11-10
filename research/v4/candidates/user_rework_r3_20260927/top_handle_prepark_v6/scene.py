"""Common normal joint initialization clear of the relocated appliance."""
from unittest.mock import patch
from research.v4.candidates.user_rework_r3_20260927.top_handle_v3 import scene as parent
PARK=(1.01078149584495,-0.03203186173024337,-2.2149888801840696,-2.029182324364757,0.5514647376956447,1.2063524542781903,-1.7372411122190972)
class PreparkR3Scene(parent.TopHandleR3Scene):
 layout_version='r3-front-facing-real-handle12cm-riser-preinitialized-left-park-v6'
 def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
  with patch.object(parent,'NEUTRAL_QPOS',PARK):
   super().__init__(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
  self.reset_protocol=dict(version='r3-common-safe-left-joint-initialization-v6',joint_qpos_rad=list(PARK),postconstructor_control_count=0,note='Original robosuite joint initialization before object reset, not a post-reset teleport. No real reset controls occurred in this carrier.')
