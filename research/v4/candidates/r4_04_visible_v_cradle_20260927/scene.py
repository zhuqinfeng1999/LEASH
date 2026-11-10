"""Normal two-face cradle, original actual upright loop pillow."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from research.v4.candidates.r4_04_visible_pull_loop_20260927 import scene as loop
from research.v4.candidates.r4_04_upright_pillow_fullsolid_z_20260927 import scene as full
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];SPEC=json.loads((HERE/'structure.json').read_text());LOOP_FACTORY=loop.loop_factory

def cradle_factory():
 base=LOOP_FACTORY()
 class CradleEnv(base):
  def _load_model(self):
   super()._load_model()
   # Parent full.make_env adds its rack after this method returns. Replace only
   # that generator in the Scene wrapper, so visible/collision faces match.
 return CradleEnv

def make_cradle_env():
 base=LOOP_FACTORY()
 class CradleEnv(base):
  def _load_model(self):
   super()._load_model()
   ET.SubElement(self.model.asset,'texture',name='r404_cradle_wood_tex',type='2d',builtin='checker',rgb1='.47 .28 .13',rgb2='.52 .32 .16',width='64',height='64')
   ET.SubElement(self.model.asset,'material',name='r404_cradle_oiled_wood',texture='r404_cradle_wood_tex',texrepeat='12 1',specular='.12',shininess='.12')
   b=ET.SubElement(self.model.worldbody,'body',name='r404_upright_wood_rack',pos='0 0 .9')
   # V faces top surfaces are actual tangent planes z=.015+abs(y+.055).
   parts=[('deck',(0,-.055,.0075),(.325,.120,.0075),0.)]
   for side in(1.,-1.):
    n=np.array([0.,side,1.])/np.sqrt(2);face=np.array([0.,-.055-side*.065,.080]);center=face-n*.0075;parts.append(('front_panel'if side>0 else'back_panel',center,(.325,.06363961030678927,.0075),-side*45.))
   for name,pos,size,angle in parts:
    quat=np.roll(Rotation.from_euler('x',angle,degrees=True).as_quat(),1)
    for vis in(False,True):ET.SubElement(b,'geom',name='r404_upright_wood_'+name+('_vis'if vis else'_col'),type='box',pos=' '.join(map(str,pos)),quat=' '.join(map(str,quat)),size=' '.join(map(str,size)),material='r404_cradle_oiled_wood',group='1'if vis else'0',contype='0'if vis else'1',conaffinity='0'if vis else'1',friction='.9 .005 .0001',solref='.02 1',**({'mass':'0'}if vis else{'density':'650'}))
 return CradleEnv

class Scene(loop.Scene):
 protocol='R404_VISIBLE_NORMAL_TWO45DEG_WOOD_CRADLE_LOOP_UNCHANGED_PILLOW_ASSIST_V1'
 def __init__(self,*a,**kw):
  # Avoid adding a second rack while retaining the original fullsolid and loop
  # initialization. Factory is only active during constructor compilation.
  with patch.object(full,'make_env',make_cradle_env):super().__init__(*a,**kw)
  try:
   sim=self.backend.sim;obj=self.backend.objects_dict[self.item_name];q=sim.data.get_joint_qpos(obj.joints[0]).copy();before=q.copy();q[2]=SPEC['body_world_z_m'];sim.data.set_joint_qpos(obj.joints[0],q);sim.data.set_joint_qvel(obj.joints[0],np.zeros(6));sim.forward();self.initial_item_z=float(q[2]);self.cradle_initial=dict(before_body_mesh_floor_qpos=before.tolist(),initial_tangent_supported_qpos=q.tolist(),actual_shape_support_source='structure.json',reset_before_controls=True,scene_geometry_changed_not_equivalent=True)
   if self.backend.use_camera_obs:self.raw=self.backend._get_observations(force_update=True)
  except BaseException:super().close();raise
