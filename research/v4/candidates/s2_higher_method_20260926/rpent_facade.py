"""Thin S2 calibration/latch facade; private dose/poses never enter public packets."""
from __future__ import annotations
import numpy as np
from research.v4.accelerated.observation import public_observation
from research.v4.candidates.s2_scene_20260924.scene import INSTRUCTIONS

class S2RPentFacade:
    def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=0,
                 diagnostic_scene_factory=None):
        if diagnostic_scene_factory is None:
            from research.v4.candidates.s2_dissolution_boundary_20260926.scene import DissolutionSafeScene
            factory=DissolutionSafeScene
        else:factory=diagnostic_scene_factory # CPU test only; runner cannot supply this
        self._scene=factory(seed=seed,size=480,width=640,grasp_assist=True,
            camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
        self.frame_id=0
        self._scene.start_round(1)
    def __getattr__(self,name):return getattr(self._scene,name)
    @property
    def sim(self):return self._scene.env.sim
    @property
    def backend(self):return self._scene.env.env
    @property
    def round_number(self):return self._scene.round_index
    @property
    def instruction(self):return INSTRUCTIONS[self.round_number-1]
    def advance_round(self):return self._scene.start_round(self.round_number+1)
    def public_grasp_latched(self):return bool(self._scene.grasp_assist.held)
    def capture_agent(self):
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix,get_camera_intrinsic_matrix
        source=self._scene.capture_agent();self.frame_id+=1;rendered={}
        for camera,frame in source['cameras'].items():
            rendered[camera]={**frame,'intrinsics':get_camera_intrinsic_matrix(self.sim,camera,480,640),
                'extrinsics':get_camera_extrinsic_matrix(self.sim,camera),'frame_id':self.frame_id}
        robot={k:source[k] for k in ('eef_pos','eef_quat','gripper_qpos')}
        packet=public_observation(rendered,robot,frame_id=self.frame_id)
        packet['robot_state']['grasp_latched']=self.public_grasp_latched()
        packet.update(robot)
        return packet
