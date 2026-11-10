"""One natural supporting-finger extension; unchanged full-person T6 physics."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from research.v4.candidates.e3_04_soup_bowl_20260926 import scene as previous
from research.v4.candidates.e3_relaxed_initial_person_20260926 import scene as relaxed
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTRUCTION=previous.INSTRUCTION
class SupportingPerson(relaxed.RelaxedPerson):
    supporting_pip_extension_deg=15.
    def pose(self,hand,right_hand=None):
        transforms=super().pose(hand,right_hand=right_hand)
        current=self.right_hand if right_hand is None else np.asarray(right_hand)
        if current is None or self.right_rest_final is None:return transforms
        delta=np.asarray(self.right_rest_final)-self.right_rest_start
        progress=float(np.clip(np.dot(np.asarray(current)-self.right_rest_start,delta)/max(float(delta@delta),1e-12),0.,1.))
        rotation=Rotation.from_euler('y',self.supporting_pip_extension_deg*progress,degrees=True).as_matrix()
        for finger in range(2,6):
            joint=f'finger{finger}-2.R';r,t=transforms[joint];pivot=r@self.head[joint]+t
            for name,(r,t)in tuple(transforms.items()):
                node=name
                while node is not None and node!=joint:node=self.bones[node]['parent']
                if node==joint:transforms[name]=(rotation@r,pivot+rotation@(t-pivot))
        return transforms
class Scene(previous.Scene):
    protocol_version='e3-04-real-whole-arm-supporting-pip-extension15-v42-v3'
    def __init__(self,*args,**kwargs):
        with patch.object(relaxed,'RelaxedPerson',SupportingPerson):super().__init__(*args,**kwargs)
