"""Root-selected sole whole-crib robot-side layout, no runtime relocation."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ..r402_lowered_rail_v2 import scene as parent
from research.v4.accelerated import tabletop
HERE=Path(__file__).resolve().parent;ROOT=parent.ROOT
class Scene(parent.Scene):
 def __init__(self,*args,**kwargs):
  original=tabletop.sample_slot
  def shifted(slot,seed,object_index=0):
   xyz,yaw=original(slot,seed,object_index=object_index)
   if slot=='R':xyz=xyz+np.array([-.56,0.,0.])
   return xyz,yaw
  with patch.object(tabletop,'sample_slot',shifted):super().__init__(*args,**kwargs)
