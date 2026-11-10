"""One visible stitched fabric loop on same upright rigid pillow and rack."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.candidates.r4_04_upright_pillow_fullsolid_z_20260927 import scene as full
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
SPEC=json.loads((HERE/'anchor.json').read_text());POINT=np.asarray(SPEC['point_body_m']);SOURCE_FACTORY=full.ORIGINAL_FACTORY

def loop_factory():
 base=SOURCE_FACTORY()
 class LoopEnv(base):
  def _load_model(self):
   super()._load_model()
   ET.SubElement(self.model.asset,'texture',name='r404_pull_loop_cotton_tex',type='2d',builtin='checker',rgb1='.68 .63 .51',rgb2='.75 .70 .58',width='64',height='64')
   ET.SubElement(self.model.asset,'material',name='r404_pull_loop_cotton',texture='r404_pull_loop_cotton_tex',texrepeat='8 8',specular='.04',shininess='.02')
   bodies=[b for b in self.model.worldbody.iter('body')if b.find("geom[@name='adult_pillow_1_pillow_body_collision']")is not None]
   if len(bodies)!=1:raise RuntimeError('Actual same pillow root body not uniquely found')
   body=bodies[0];x,y,z=SPEC['anchor_body_m'];parts=[('left_leg',(x-.009,y+.0175,z),(.002,.019,.001)),('right_leg',(x+.009,y+.0175,z),(.002,.019,.001)),('upper_bridge',(x,y+.0345,z),(.011,.002,.001))]
   for name,pos,size in parts:
    for vis in(False,True):ET.SubElement(body,'geom',name='adult_pillow_1_pull_loop_'+name+('_vis'if vis else'_collision'),type='box',pos=' '.join(map(str,pos)),size=' '.join(map(str,size)),material='r404_pull_loop_cotton',group='1'if vis else'0',contype='0'if vis else'1',conaffinity='0'if vis else'1',mass='0',friction='.9 .005 .0001',solref='.02 1')
 return LoopEnv

class Scene(full.Scene):
 protocol='R404_UPRIGHT_ORIGINAL_ADULT_PILLOW_VISIBLE_STITCHED37MM_LOOP_ASSIST_REGION_V1'
 def __init__(self,*a,**kw):
  with patch.object(full,'ORIGINAL_FACTORY',loop_factory):super().__init__(*a,**kw)
  try:
   sim=self.backend.sim;sim.model.site_pos[sim.model.site_name2id('v42_adult_pillow_1_grasp_zone')]=POINT;sim.forward();m=sim.model._model;bid=m.body(self.backend.objects_dict[self.item_name].root_body).id
   self.visible_loop_metadata=dict(authored_CC0=True,rigid_attached_pillowcase_loop=True,body_zone=POINT.tolist(),cloth_material_thickness_m=.002,visible_solid_loop_bounding_dimensions_m=[.022,.0395,.002],pillow_total_mass_kg=float(m.body_mass[bid]),pillow_total_inertia=m.body_inertia[bid].tolist(),mass_model='Original explicit650g/.008,.016,.022 diagonal inertia retained as pillow+case assembly. Loop geometry mass0 represents redistributed <1g case fabric; full dynamic inertia is an authored approximation, not physical equivalence to old geometry.',no_online_private_zone=True)
   if self.backend.use_camera_obs:self.raw=self.backend._get_observations(force_update=True)
  except BaseException:super().close();raise
