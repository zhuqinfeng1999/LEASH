"""Own CC0 bead box + readable card; normal full crib via explicit config only."""
import copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.accelerated import tabletop
from research.v4.candidates.r4_05_identity_uv_fix_20260926 import scene as readable
from research.v4.candidates.r4_05_inclined_identity_v2_20260926 import scene as original
from ..r402_drop_side_v5 import scene as crib
HERE=Path(__file__).resolve().parent;ROOT=crib.ROOT

def configuration():
 c=copy.deepcopy(readable.configuration())
 c['crib'].update(xml=str(crib.XML),initial_z_m=.64,footprint_half_xy_m=[.632,.332],inside_half_xy_m=[.60,.30],inside_bottom_z_m=.040,inside_top_z_m=.45)
 c['layout']['crib_dy_m']=0.
 c['license_status']='Own authored bead-box/identity CC0 retained; normal fullsize crib and existing CC0 wood source reused, aggregate publication review remains root-owned.'
 return c
class Scene(original.T1R4MagneticBoxScene):
 def __init__(self,*a,**kw):
  sample=tabletop.sample_slot
  def own_slot(slot,seed,object_index=0):
   xyz,yaw=sample(slot,seed,object_index=object_index)
   if slot=='R':xyz=xyz+np.array([-.40,.66,-.26])
   return xyz,yaw
  with patch.object(original,'configuration',configuration),patch.object(tabletop,'sample_slot',own_slot):super().__init__(*a,**kw)
  self.furniture_access_version='own-R405-normal-fullsize-drop-side-v1'
