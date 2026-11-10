"""Own unchanged210g coin-dish source with qualified normal-size floor crib."""
from pathlib import Path
from unittest.mock import patch
import copy,numpy as np
from research.v4.candidates.r4_06_flat_underbase_v2_20260926 import scene as parent
from research.v4.accelerated import tabletop
from ..r402_drop_side_v5.scene import XML
ROOT=parent.ROOT
BASE_CONFIG=parent.configuration
def configuration():
 c=copy.deepcopy(BASE_CONFIG());c['crib'].update(xml=str(XML),initial_z_m=.64,footprint_half_xy_m=[.632,.332],inside_half_xy_m=[.60,.30],inside_bottom_z_m=.040,inside_top_z_m=.45);c['layout']['crib_dy_m']=0.;return c
class Scene(parent.T1R4CoinDishScene):
 def __init__(self,*a,**kw):
  sample=tabletop.sample_slot
  def own_slot(slot,seed,object_index=0):
   xyz,yaw=sample(slot,seed,object_index=object_index)
   if slot=='R':xyz=xyz+np.array([-.40,.66,-.26])
   return xyz,yaw
  with patch.object(parent,'configuration',configuration),patch.object(tabletop,'sample_slot',own_slot):super().__init__(*a,**kw)
  m=self.backend.sim.model._model;d=self.backend.sim.data._data
  self.initial_contact_witness=[dict(geoms=[m.geom(int(i)).name or''for i in(c.geom1,c.geom2)],distance_m=float(c.dist))for c in d.contact[:d.ncon]]
  bad=[c for c in self.initial_contact_witness if c['distance_m']<-.001 and any(n.startswith('crib_1_')for n in c['geoms'])and any(n.startswith(('robot0_','gripper0_'))or n=='table_collision'for n in c['geoms'])]
  if bad:self.close();raise RuntimeError('Initial full crib/table/robot macro collision')
  self.initial_coin_dish_qpos=self.backend.sim.data.get_joint_qpos(self.backend.objects_dict[self.item_name].joints[0]).copy().tolist()
  self.initial_normal_crib_qpos=self.backend.sim.data.get_joint_qpos(self.backend.objects_dict[self.target_name].joints[0]).copy().tolist()
