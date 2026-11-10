"""Substantive front-source/back-container layout, shared real physics."""
from dataclasses import replace
from unittest.mock import patch
from research.v4.candidates.user_rework_support_20260927.i1_layout_v2 import scene as old
common=old.common
NOMINAL_XY={'paper_bin_1':(-.28,-.26),'blunt_tool_1':(.07,-.08),'soup_can_1':(.20,-.18)}
BINDINGS=tuple(replace(b,slot='B',xy_offset=(0.,-.26))if b.name=='paper_bin_1'else
               replace(b,slot='C',xy_offset=(.15,-.08))if b.name=='blunt_tool_1'else
               replace(b,slot='C',xy_offset=(.28,-.18))if b.name=='soup_can_1'else b for b in old.BINDINGS)
def create(seed,*,camera_observations=False,render_gpu_device_id=-1):
    with patch.object(common,'I1StraightRightEnv',old.LayoutV2Env),patch.object(common,'BINDINGS',BINDINGS):
        return common.create(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
