"""Render-disabled transport of the unchanged S2 scene for CPU scripts only.

This is not an online scene and exposes no planner/reviewer interfaces.
Only renderer flags differ from OffScreenRenderEnv; the robot/controller,
MJCF objects, seeds, settling, grasp assist and dose events are unchanged.
"""
from unittest.mock import patch
import os
from research.v4.candidates.s2_scene_20260924.scene import Scene,ROOT

def cpu_factory(**kwargs):
    from libero.libero.envs.env_wrapper import ControlEnv
    return ControlEnv(**kwargs,use_camera_obs=False,has_renderer=False,
                      has_offscreen_renderer=False)

class CpuScene(Scene):
    def __init__(self,*args,**kwargs):
        os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'configs/runtime/libero'))
        with patch('libero.libero.envs.OffScreenRenderEnv',cpu_factory):
            super().__init__(*args,**kwargs)
        if self.env.env.has_offscreen_renderer or self.env.env.use_camera_obs:
            raise RuntimeError('S2 CPU scene unexpectedly enabled rendering')
