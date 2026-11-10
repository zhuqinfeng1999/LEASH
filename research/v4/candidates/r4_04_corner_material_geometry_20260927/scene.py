"""Original actual adult pillow; real corner, fixed jitter-preserving placement."""
import copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.candidates.r4_04_real_corner_zone_v2_20260926 import scene as original
from research.v4.candidates.r4_crib_doll_raised3mm_20260926.candidate import XML
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
ORIGINAL_CONFIGURATION=original.configuration
CORNER=(-.248,-.168,.055);SOURCE_SHIFT_X=-.160
def configuration():
 c=copy.deepcopy(ORIGINAL_CONFIGURATION());c['item']['xml']=str(original.HERE/c['item']['xml']);c['item']['grasp_local']=list(CORNER);c['crib']['xml']=str(XML);return c
class Scene(original.T1R4AdultPillowScene):
 protocol='R404_ORIGINAL_REAL_CLOTH_CORNER_FIXED_XSHIFT160MM_COMMON_JITTER_FAITHFUL19_RAISED3MM'
 def __init__(self,*args,**kw):
  with patch.object(original,'configuration',configuration):super().__init__(*args,**kw)
  try:
   sim=self.backend.sim;obj=self.backend.objects_dict[self.item_name];q=sim.data.get_joint_qpos(obj.joints[0]).copy();before=q.copy();q[0]+=SOURCE_SHIFT_X;sim.data.set_joint_qpos(obj.joints[0],q);sim.data.set_joint_qvel(obj.joints[0],np.zeros(6));sim.forward();self.initial_item_z=float(q[2])
   m=sim.model._model;bid=m.body(obj.root_body).id;baby=[m.geom(i).name for i in range(m.ngeom)if(m.geom(i).name or'').startswith('crib_1_doll_')and m.geom_group[i]==0]
   if len(baby)!=19:raise RuntimeError('current faithful infant collider protocol missing')
   self.candidate_initial=dict(original_sampled_source_qpos=before.tolist(),initial_candidate_qpos=q.tolist(),fixed_shift_x_m=SOURCE_SHIFT_X,common_xy_yaw_jitter_preserved=True,source_mass_kg=float(m.body_mass[bid]),source_inertia_kg_m2=m.body_inertia[bid].tolist(),faithful_infant_collision_geoms=baby,physical_shape_unchanged=True,private_grasp_region_only=CORNER,initial_only_before_any_control=True)
   if self.backend.use_camera_obs:self.raw=self.backend._get_observations(force_update=True)
  except BaseException:super().close();raise
