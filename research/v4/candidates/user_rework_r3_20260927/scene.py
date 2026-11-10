"""Whole-device/source layout revision only; common R3 physics/API retained."""
from dataclasses import replace
from unittest.mock import patch
import math
import numpy as np
from research.v4.accelerated import tabletop
from research.v4.candidates.r3_v42_tabletop_20260925 import scene as original
from research.v4.candidates.r3_v42_visibility_20260926.scene_lower_aim import NEUTRAL_QPOS
SOURCE_NOMINAL_XY=(-.22,-.20)
OVEN_NOMINAL_XY=(.13,.14)
OVEN_NOMINAL_YAW=-math.pi/2
class FrontFixtureEnv(original._R3FixtureEnv):
 def _reset_internal(self):
  super()._reset_internal()
  obj=self.objects_dict[original.OVEN]
  q=np.asarray(self.sim.data.get_joint_qpos(obj.joints[0]),float)
  delta=2*math.atan2(q[6],q[3]);yaw=OVEN_NOMINAL_YAW+delta
  q[3:]=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
  self.sim.data.set_joint_qpos(obj.joints[0],q)
  self.sim.forward()
class FrontR3Scene(original.R3TabletopFacade):
 task_id='R3-01'
 layout_version='r3-front-facing-whole-device-near-source-side-knob-v1'
 def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
  bindings=tuple(replace(b,xy_offset=(SOURCE_NOMINAL_XY[0]-tabletop.SLOTS['L'].xyz[0],SOURCE_NOMINAL_XY[1]-tabletop.SLOTS['L'].xyz[1]))if b.name==original.ITEM else replace(b,xy_offset=(OVEN_NOMINAL_XY[0]-tabletop.SLOTS['R'].xyz[0],OVEN_NOMINAL_XY[1]-tabletop.SLOTS['R'].xyz[1]))for b in original._BINDINGS)
  with patch.object(original,'_BINDINGS',bindings),patch.object(original,'_R3FixtureEnv',FrontFixtureEnv),patch.object(tabletop,'PARK_QPOS',NEUTRAL_QPOS):
   super().__init__(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
