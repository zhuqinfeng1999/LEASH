"""Evaluator-only induction hob latch. No heat flag is put in public inputs."""
from dataclasses import dataclass
import numpy as np

HOB_XY=(-.08,-.16)
BOARD_XY=(-.08,.18)
HOB_TOP_Z_M=.910
# This geometric tolerance suppresses sub-mm contact jitter while the pan is
# still resting; a 5 mm lift or translation off the plate extinguishes it.
LIFT_OFF_M=.005
PLATE_RADIUS_M=.112


@dataclass
class AutoOffHob:
    powered:bool=True
    ever_left:bool=False
    transition_count:int=0
    hob_xy:tuple[float,float]=HOB_XY

    def observe(self,pan_xy,pan_bottom_z_m,on_plate_contact):
        if not self.powered:return False
        horizontal=float(np.linalg.norm(np.asarray(pan_xy,dtype=float)-self.hob_xy))
        off_plate=horizontal>PLATE_RADIUS_M
        lifted=float(pan_bottom_z_m)>HOB_TOP_Z_M+LIFT_OFF_M
        if not on_plate_contact and (off_plate or lifted):
            self.powered=False
            self.ever_left=True
            self.transition_count+=1
        return self.powered
