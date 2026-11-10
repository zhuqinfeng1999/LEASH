"""Unchanged genuine cloth corner and analytically whole-supported common layout."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.candidates.r4_04_corner_material_geometry_20260927 import scene as previous
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
SIDE_R=((0.,1.,0.),(0.,0.,1.),(1.,0.,0.))
class Scene(previous.original.T1R4AdultPillowScene):
 protocol='R404_ORIGINAL_CLOTH_Z_PINCH_ANALYTIC_WHOLE_SUPPORT_COMMON_XY20MM_YAW10DEG_FAITHFUL19_RAISED3MM'
 def __init__(self,*a,**kw):
  with patch.object(previous.original,'configuration',previous.configuration):super().__init__(*a,**kw)
  try:
   sim=self.backend.sim;obj=self.backend.objects_dict[self.item_name];q=sim.data.get_joint_qpos(obj.joints[0]).copy();old=q.copy();p=json.loads((HERE/'proposal.json').read_text());q[:2]+=p['source_shift_xy_m'];sim.data.set_joint_qpos(obj.joints[0],q);sim.data.set_joint_qvel(obj.joints[0],np.zeros(6));sim.forward();self.initial_item_z=float(q[2]);m=sim.model._model;bid=m.body(obj.root_body).id
   baby=[m.geom(i).name for i in range(m.ngeom)if(m.geom(i).name or'').startswith('crib_1_doll_')and m.geom_group[i]==0]
   if len(baby)!=19:raise RuntimeError('faithful19 infant colliders missing')
   self.candidate_initial=dict(original_sampled_qpos=old.tolist(),source_qpos=q.tolist(),nominal_common_supported_xy_m=p['source_nominal_xy_m'],constant_shift_xy_m=p['source_shift_xy_m'],all_common_xy_yaw_jitter_preserved=True,faithful_infant_colliders=baby,mass_kg=float(m.body_mass[bid]),inertia_kg_m2=m.body_inertia[bid].tolist(),shape_visual_collision_unchanged=True)
   if self.backend.use_camera_obs:self.raw=self.backend._get_observations(force_update=True)
  except BaseException:super().close();raise
