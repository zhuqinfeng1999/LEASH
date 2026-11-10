"""Bounded natural complete-arm arc correction over the carrot, same .85s event."""
from pathlib import Path
import numpy as np
from research.v3.tasks.e1.private_dynamics import smoothstep
from research.v4.candidates.e1_02_natural_support_v2_20260926.scene import E1CleaverBoardScene as Previous
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PROTOCOL='e1-02-cleaver-cuttingboard-natural-arm-arc-t5-assisted-v42-v3'
class E1CleaverBoardScene(Previous):
 protocol_version=PROTOCOL
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  original_update=self.person.update
  def physical_full_person_update(sim,left_hand):
   if self.trigger_time_s is not None:
    elapsed=float(sim.data.time)-self.trigger_time_s
    blend=float(smoothstep(elapsed/self.hand_duration_s))
    lift=.060*max(0.,float(np.sin(np.pi*blend)))
    self.hand=self.hand_start+blend*(self.hand_final-self.hand_start)+np.array([0.,0.,lift])
    self.person.right_hand=self.hand.copy()
   # Existing full-person rig solves shoulder/elbow/wrist and all bone poses;
   # update happens before the actual controller step, not after rendering.
   original_update(sim,left_hand)
  self.person.update=physical_full_person_update
