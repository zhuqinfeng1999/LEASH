"""Root-authorized task-fidelity correction: real fingers enter unchanged box."""
from pathlib import Path
import numpy as np
from research.v4.candidates.e1_05_pre_lift_arm_v2_20260926.scene import E1GlassBottleBoxScene as Previous
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PROTOCOL='e1-05-glass-flask-cardboard-box-inside-right-hand-t5-assisted-v42-v3'
class E1GlassBottleBoxScene(Previous):
 protocol_version=PROTOCOL
 def _target_hand_pose(self):return super()._target_hand_pose()-np.array([0,0,.015])
