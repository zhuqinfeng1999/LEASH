"""One causal layout correction: front-left bin clear of right arm, device clear of basket."""
from dataclasses import replace
from unittest.mock import patch
from research.v4.candidates.user_rework_support_20260927 import scene as first
common=first.common
MICROWAVE_CENTER=(-.22,.29,.9);BIN_CENTER=(.25,-.16,.9)
BINDINGS=tuple(replace(b,xy_offset=(-.07,-.16))if b.name=='paper_bin_1'else b for b in first.BINDINGS)
class LayoutV2Env(first.SupportEnv):
 @staticmethod
 def person_builder(arena):
  first.SupportEnv.person_builder(arena)
  arena.root.find(".//body[@name='i1_microwave_fixture']").set('pos',' '.join(map(str,MICROWAVE_CENTER)))
def create(seed,*,camera_observations=False,render_gpu_device_id=-1):
 with patch.object(common,'I1StraightRightEnv',LayoutV2Env),patch.object(common,'BINDINGS',BINDINGS):
  return common.create(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
