"""One justified pose correction: the full person's fingers support the carrot.

Preserve all E1-02 asset dimensions, common OSC/assist, event timing/collision
masks and risk predicates. No standalone hand or runtime risk relaxation.
"""
from pathlib import Path
import numpy as np
from research.v4.candidates.e1_02_cleaver_board_20260926.scene import E1CleaverBoardScene as Previous
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PROTOCOL='e1-02-cleaver-cuttingboard-natural-support-t5-assisted-v42-v2'
class E1CleaverBoardScene(Previous):
 protocol_version=PROTOCOL
 def _target_hand_pose(self):return super()._target_hand_pose()-np.array([0.,0.,.046])
