"""Distinct normal closer destination at reset; no runtime pose assignment."""
from pathlib import Path
import numpy as np
from research.v4.candidates.p1_06_high_neck_v2_20260926.scene import P1WineBucketFacade as Previous,INSTRUCTION
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
class P1WineBucketFacade(Previous):
 protocol_version='p1-06-nearer-bucket-rim-clearance-v3'
 def _place_e1_layout_at_reset(self):
  super()._place_e1_layout_at_reset()
  sim=self.backend.sim;target=self.backend.objects_dict[self.target_name]
  pose=np.asarray(sim.data.get_joint_qpos(target.joints[0])).copy();pose[0]-=.080
  sim.data.set_joint_qpos(target.joints[0],pose);sim.data.set_joint_qvel(target.joints[0],np.zeros(6));sim.forward()
