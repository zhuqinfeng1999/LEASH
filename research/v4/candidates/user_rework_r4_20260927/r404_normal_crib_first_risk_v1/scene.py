"""Own upright650g/V-cradle/240reset; only normal furniture config changes."""
import copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.accelerated import tabletop
from research.v4.candidates.r4_04_v_cradle_post_orientation_20260927 import scripts as parent
from research.v4.candidates.r4_04_corner_material_geometry_20260927 import scene as corner
from ..r402_drop_side_v5 import scene as crib
HERE=Path(__file__).resolve().parent;ROOT=crib.ROOT
ORIGINAL_CORNER_CONFIG=corner.configuration

def configuration():
 c=copy.deepcopy(ORIGINAL_CORNER_CONFIG())
 c['crib'].update(xml=str(crib.XML),initial_z_m=.64,footprint_half_xy_m=[.632,.332],inside_half_xy_m=[.60,.30],inside_bottom_z_m=.040,inside_top_z_m=.45)
 c['layout']['crib_dy_m']=0.;return c

class Scene(parent.previous.SettledScene):
 def __init__(self,*a,**kw):
  sample=tabletop.sample_slot
  def slot(name,seed,object_index=0):
   xyz,yaw=sample(name,seed,object_index=object_index)
   if name=='R':xyz=xyz+np.array([-.40,.66,-.26])
   return xyz,yaw
  with patch.object(corner,'configuration',configuration),patch.object(tabletop,'sample_slot',slot):super().__init__(*a,**kw)
  try:
   raw=parent.geometry(self)
   source_key=bool(all(v['zone']['in_frame']and(v['seam_ray_hit']or'').startswith('adult_pillow_1_')for v in raw['views'].values())and raw['views']['agentview']['pillow']['in_frame'])
   physical=bool(raw['whole_rack_pillow_table_XY']and not raw['blockers']and not raw['deep_contacts']and source_key)
   self.actual_reset_geometry={**raw,'legacy_every_infant_each_public_view_pass':raw['passed'],'raw_legacy_public_key_visibility':raw['public_key_visibility'],'source_key_visibility_unchanged_required':source_key,'physical_reset_geometry_pass':physical,'public_native_visibility_pending':True,'passed':physical,'geometry_companion_scope':'Normal-furniture physical mechanics only; wrist need not contain full infant/crib. Legacy projection/raw remains.'}
   if not physical:raise RuntimeError('Own actual normal-crib V-cradle physical reset failure')
  except BaseException:super().close();raise
  self.furniture_access_version='R404-own-normal-crib-V-cradle-240reset-v1'
