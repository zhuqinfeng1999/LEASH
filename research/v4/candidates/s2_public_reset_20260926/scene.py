"""S2 candidate: fixed initial Panda joint pose, original cameras and assets."""
from pathlib import Path
from unittest.mock import patch
import json,os
import numpy as np
from research.v4.candidates.s2_scene_20260924.scene import Scene as OriginalScene,ROOT

HERE=Path(__file__).resolve().parent
PARK_QPOS=np.asarray(json.loads((HERE/'selected_pose.json').read_text())['initial_joint_qpos_rad'],float)

def parked_env_factory(*,camera_observations=False,render_gpu_device_id=0,**kwargs):
    from libero.libero.envs.env_wrapper import ControlEnv
    kwargs['render_gpu_device_id']=render_gpu_device_id
    env=ControlEnv(**kwargs,use_camera_obs=camera_observations,
                   has_offscreen_renderer=camera_observations,has_renderer=False)
    env.env.robots[0].init_qpos=PARK_QPOS.copy()
    env.env.robot_configs[0]['initial_qpos']=PARK_QPOS.copy()
    return env

class PoseScene(OriginalScene):
    def __init__(self,seed=0,size=480,width=640,grasp_assist=True,*,camera_observations=False,render_gpu_device_id=0):
        os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'configs/runtime/libero'))
        def factory(**kwargs):
            kwargs['render_gpu_device_id']=render_gpu_device_id
            return parked_env_factory(camera_observations=camera_observations,
                **kwargs)
        with patch('libero.libero.envs.OffScreenRenderEnv',factory):
            super().__init__(seed=seed,size=size,width=width,grasp_assist=grasp_assist)
        self.camera_observations=bool(camera_observations)
        if not camera_observations and self.env.env.has_offscreen_renderer:
            raise RuntimeError('CPU pose scene unexpectedly has renderer')
