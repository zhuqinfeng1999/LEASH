"""Final authorized sole task-fidelity correction: wrist endpoint5mm lower."""
from pathlib import Path
import numpy as np
from research.v4.candidates.e1_03_flat_base_entry_v4_20260926.scene import E1ScrewdriverToolboxScene as Previous
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PROTOCOL='e1-03-screwdriver-toolbox-real-inside-right-hand-t5-assisted-v42-v5'
class E1ScrewdriverToolboxScene(Previous):
 protocol_version=PROTOCOL
 def _target_hand_pose(self):return super()._target_hand_pose()-np.array([0,0,.005])
