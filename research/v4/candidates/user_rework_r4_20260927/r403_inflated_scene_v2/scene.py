"""Explicit inflated R403 version over its own existing common T1 constructor."""
import copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.accelerated import tabletop
from research.v4.candidates.r4_03_deflated_balloon_t1_20260926 import scene as original
from ..r403_inflated_asset_v1 import make_asset as asset
from ..r402_drop_side_v5 import scene as crib
HERE=Path(__file__).resolve().parent
ROOT=original.ROOT
ORIGINAL_CONFIGURATION=original.configuration
PROTOCOL='R403_USER_INFLATED_NORMAL_CRIB_CANDIDATE_V2'
def configuration():
 # Validate the original registered English/role first, explicitly version the
 # user's subsequent inflated-asset request rather than altering the registry.
 c=copy.deepcopy(ORIGINAL_CONFIGURATION())
 c['item'].update(name='inflated_balloon_1',category='rigid_inflated_latex_balloon',display_name='Inflated small tied latex balloon',xml=str(asset.HERE/'inflated_balloon.xml'),grasp_local=[.140,0,.122],z_offset=.016,initial_z_m=.916,center_offset_z_m=.122,bottom_offset_z_m=.002,footprint_half_xy_m=[.154,.090],mass_kg=.008)
 c['board']['xml']=str(original.HERE/'assets/board.xml')
 c['crib'].update(xml=str(crib.XML),initial_z_m=.64,footprint_half_xy_m=[.632,.332],inside_half_xy_m=[.60,.30],inside_bottom_z_m=.040,inside_top_z_m=.45)
 c['layout']['crib_dy_m']=0.
 c['protocol']=PROTOCOL
 c['physics_proxy']='Authored rigid inflated latex volume, real neck solid, no elasticity/burst/buoyancy claim; original region tolerance unchanged.'
 c['task_version_note']='User-requested inflated balloon supersedes original uninflated asset for this future version only.'
 return c
class Scene(original.T1R4BalloonScene):
 item_name='inflated_balloon_1'
 protocol=PROTOCOL
 def __init__(self,*a,**kw):
  sample=tabletop.sample_slot
  def own_slot(slot,seed,object_index=0):
   xyz,yaw=sample(slot,seed,object_index=object_index)
   if slot=='R':xyz=xyz+np.array([-.40,.66,-.26])
   return xyz,yaw
  with patch.object(original,'configuration',configuration),patch.object(tabletop,'sample_slot',own_slot):super().__init__(*a,**kw)
  self.source_scene_xml=self.backend.model.get_xml()
