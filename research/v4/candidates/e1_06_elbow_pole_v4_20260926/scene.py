"""One natural whole-arm elbow-pole correction; old physics and wrist path frozen."""
from pathlib import Path
from types import MethodType
import numpy as np
from research.v3.tasks.e1.private_dynamics import smoothstep
from research.v4.candidates.e1_06_flat_contact_raised_arm_v3_20260926.scene import E1GlassDishpanScene as Previous
from .person_pose import pose
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PROTOCOL='e1-06-natural-raised-elbow-wholearm-t5-assisted-v42-v4'
def elbow_pole(phase):
 blend=float(smoothstep(float(np.clip(phase/.35,0.,1.))))
 return (1-blend)*np.array([1.,.5,-.7])+blend*np.array([0.,.5,1.])
def hand_at(scene,elapsed):
 phase=float(np.clip(elapsed/scene.hand_duration_s,0.,1.));blend=float(smoothstep((phase-.25)/.75));raise_blend=float(smoothstep(phase/.25));lower_blend=float(smoothstep((1.-phase)/.30));hand=scene.hand_start+blend*(scene.hand_final-scene.hand_start)+[0,0,.200*min(raise_blend,lower_blend)]
 if elapsed>scene.hand_duration_s:
  t=elapsed-scene.hand_duration_s;scale=float(smoothstep(t/.35));hand+=scale*np.array([.006*np.sin(2*np.pi*.8*t),.004*(1-np.cos(2*np.pi*.8*t)),0.])
 return hand
class E1GlassDishpanScene(Previous):
 protocol_version=PROTOCOL
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.person._right_elbow_pole=elbow_pole(0.);self.person.pose=MethodType(pose,self.person);old_update=self.person.update;self.elbow_pole_runtime_trace=[]
  def update(sim,left_hand):
   phase=0. if self.trigger_time_s is None else (float(sim.data.time)-self.trigger_time_s)/self.hand_duration_s
   self.person._right_elbow_pole=elbow_pole(phase)
   old_update(sim,left_hand)
   if not np.array_equal(self.person._right_elbow_pole,elbow_pole(phase)):raise RuntimeError('actual person update reset declared elbow pole')
   self.elbow_pole_runtime_trace.append(dict(sim_time_s=float(sim.data.time),phase=float(phase),actual_pole=self.person._right_elbow_pole.tolist(),expected_pole=elbow_pole(phase).tolist()))
  self.person.update=update;self.person.update(self.backend.sim,self.e1_left_rest_xyz_m);self.backend.sim.forward()
  if self.backend.use_camera_obs:self.raw=self.backend._get_observations(force_update=True)
