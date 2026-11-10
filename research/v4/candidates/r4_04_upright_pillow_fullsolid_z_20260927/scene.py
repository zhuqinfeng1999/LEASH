"""One fixed original CC0 wood rack and whole upright adult pillow."""
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from research.v4.accelerated import tabletop
from research.v4.candidates.r4_04_analytical_supported_layout_20260927.scene import Scene as ParentScene
from research.v4.candidates.r4_04_shift80_seam_copied_20260927.check import geom_points
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];POINT=np.array([-.296,0.,.055]);ORIGINAL_FACTORY=tabletop._make_env_class

def lower_support(m,center,R,g):
 kind=int(m.geom_type[g]);size=m.geom_size[g]
 if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
  mid=m.geom_dataid[g];a=m.mesh_vertadr[mid];n=m.mesh_vertnum[mid];p=np.asarray(m.mesh_vert[a:a+n])@R.T+center;z=p[:,2].min();points=p[np.abs(p[:,2]-z)<1e-7]
 elif kind==int(mujoco.mjtGeom.mjGEOM_CAPSULE):
  axis=R[:,2];signs=[-np.sign(axis[2])]if abs(axis[2])>1e-9 else[-1.,1.];points=np.array([center+sg*size[1]*axis-[0.,0.,size[0]]for sg in signs])
 elif kind==int(mujoco.mjtGeom.mjGEOM_SPHERE):points=np.array([center-[0.,0.,size[0]]])
 elif kind==int(mujoco.mjtGeom.mjGEOM_BOX):
  import itertools
  p=np.array(list(itertools.product((-1,1),repeat=3)))*size@R.T+center;z=p[:,2].min();points=p[np.abs(p[:,2]-z)<1e-7]
 elif kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER):
  rz=R[2];rad=np.linalg.norm(rz[:2]);xy=-size[0]*rz[:2]/rad if rad>1e-12 else np.zeros(2);zs=[-np.sign(rz[2])*size[1]]if abs(rz[2])>1e-9 else[-size[1],size[1]];points=np.array([center+R@np.r_[xy,z]for z in zs])
 else:raise RuntimeError('Unsupported unchanged pillow solid '+m.geom(g).name)
 return points

def make_env():
 base=ORIGINAL_FACTORY()
 class RackEnv(base):
  def _load_model(self):
   super()._load_model()
   ET.SubElement(self.model.asset,'texture',name='r404_upright_wood_tex',type='2d',builtin='checker',rgb1='.47 .28 .13',rgb2='.52 .32 .16',width='64',height='64')
   ET.SubElement(self.model.asset,'material',name='r404_upright_oiled_wood',texture='r404_upright_wood_tex',texrepeat='12 1',specular='.12',shininess='.12')
   body=ET.SubElement(self.model.worldbody,'body',name='r404_upright_wood_rack',pos='0 0 .9')
   parts=[('deck',(0,-.055,.0075),(.325,.075,.0075)),('front_rail',(0,-.1225,.025),(.325,.0075,.010)),('back_rail',(0,.0125,.025),(.325,.0075,.010)),('left_end',(-.3175,-.055,.025),(.0075,.060,.010)),('right_end',(.3175,-.055,.025),(.0075,.060,.010))]
   for name,pos,size in parts:
    for visual in (False,True):
     ET.SubElement(body,'geom',name='r404_upright_wood_'+name+('_vis'if visual else'_col'),type='box',pos=' '.join(map(str,pos)),size=' '.join(map(str,size)),material='r404_upright_oiled_wood',group='1'if visual else'0',contype='0'if visual else'1',conaffinity='0'if visual else'1',friction='.9 .005 .0001',solref='.02 1',**({'mass':'0'}if visual else{'density':'650'}))
 return RackEnv

class Scene(ParentScene):
 protocol='R404_ONE_WHOLE_X90_UPRIGHT_ADULT_PILLOW_REAL_WOOD_RACK_COPY_ONLY'
 def __init__(self,*a,**kw):
  with patch.object(tabletop,'_make_env_class',make_env):super().__init__(*a,**kw)
  try:
   sim=self.backend.sim;m,d=sim.model._model,sim.data._data;obj=self.backend.objects_dict[self.item_name];bid=m.body(obj.root_body).id
   q=sim.data.get_joint_qpos(obj.joints[0]).copy();original=q.copy();oldR=d.xmat[bid].reshape(3,3).copy();verts=geom_points(m,d,m.geom('adult_pillow_1_pillow_body_collision').id);local=(verts-d.xpos[bid])@oldR
   R=oldR@Rotation.from_euler('x',90,degrees=True).as_matrix();solid=[];visual=[]
   for g in range(m.ngeom):
    if not(m.geom(g).name or'').startswith('adult_pillow_1_'):continue
    center=R@(oldR.T@(d.geom_xpos[g]-d.xpos[bid]));gr=R@oldR.T@d.geom_xmat[g].reshape(3,3);pts=lower_support(m,center,gr,g);row=dict(name=m.geom(g).name,type=int(m.geom_type[g]),lowest_rotated_relative_z_m=float(pts[:,2].min()),lowest_support_points_relative=pts.tolist())
    if m.geom_contype[g]or m.geom_conaffinity[g]:solid.append(row)
    if m.geom_group[g]==1:visual.append(row)
   lowest=min(x['lowest_rotated_relative_z_m']for x in solid);q[0]+=.080;q[2]=.915-lowest;q[3:]=np.roll(Rotation.from_matrix(R).as_quat(),1)
   rack=m.body('r404_upright_wood_rack').id;m.body_pos[rack]=[q[0],q[1],.900];m.body_quat[rack]=original[3:]
   sim.data.set_joint_qpos(obj.joints[0],q);sim.data.set_joint_qvel(obj.joints[0],np.zeros(6));m.site_pos[m.site('v42_adult_pillow_1_grasp_zone').id]=POINT;sim.forward();self.initial_item_z=float(q[2]);self.upright_initial=dict(original_qpos=original.tolist(),pillow_qpos=q.tolist(),body_rotation_x_degrees=90,actual_mesh_minimum_rotated_relative_z_m=float((local@R.T)[:,2].min()),full_solid_minimum_rotated_relative_z_m=lowest,all_solid_height_derivation=solid,visual_minimum_rotated_relative_z_m=min(x['lowest_rotated_relative_z_m']for x in visual),one_analytical_source_z_correction_m=-(lowest-float((local@R.T)[:,2].min())),wood_deck_surface_z_m=.915,original_zone=POINT.tolist(),rack_wood_CC0_original=True,source_weld_active=False,actual_controls=0)
   if self.backend.use_camera_obs:self.raw=self.backend._get_observations(force_update=True)
  except BaseException:super().close();raise
