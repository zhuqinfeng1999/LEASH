"""One declared normal floor-crib source layout, common tabletop execution."""
import copy,json,xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.candidates.r4_02_plug_surface_v2_20260926 import scene as parent
from research.v4.candidates.r4_crib_doll_raised3mm_20260926 import candidate as baby
from research.v4.accelerated import tabletop
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4];XML=HERE/'floor_crib.xml'

def generate():
 root=ET.Element('mujoco',model='original_real_size_floor_crib')
 asset=ET.SubElement(root,'asset');ET.SubElement(asset,'texture',name='crib_wood',type='2d',file=str(ROOT/'research/upstream/polyhaven/wood_table_diff_1k.png'))
 ET.SubElement(asset,'material',name='wood',texture='crib_wood',specular='.08',shininess='.1')
 world=ET.SubElement(root,'worldbody');wrap=ET.SubElement(world,'body');body=ET.SubElement(wrap,'body',name='object')
 ET.SubElement(body,'inertial',pos='0 0 -.22',mass='25',diaginertia='3.6 4.8 5.4')
 def solid(name,pos,size,rgba=None,material=None):
  for visual in (False,True):
   kw=dict(name=name+('_visual'if visual else'_collision'),type='box',size=' '.join(map(str,size)),pos=' '.join(map(str,pos)),group='1'if visual else'0',contype='0'if visual else'1',conaffinity='0'if visual else'1',mass='0',friction='1 .005 .001',solref='.02 1')
   if material:kw['material']=material
   else:kw['rgba']=rgba or'.92 .89 .82 1'
   ET.SubElement(body,'geom',**kw)
 solid('crib_base',(0,0,.005),(.62,.32,.015),material='wood')
 solid('crib_mattress',(0,0,.025),(.6,.3,.015),rgba='.93 .90 .83 1')
 for side in(-1,1):
  for z in(.09,.435):solid(('front'if side<0 else'back')+'_rail_'+str(z),(0,side*.31,z),(.62,.012,.015),material='wood')
  for i,x in enumerate(np.linspace(-.58,.58,21)):solid(('front'if side<0 else'back')+'_slat_'+str(i),(float(x),side*.31,.2625),(.01,.01,.1575),material='wood')
  solid(('head'if side<0 else'foot')+'_panel',(side*.61,0,.24),(.012,.32,.21),material='wood')
  for y in(-.275,.275):solid('leg_'+str(side)+'_'+str(y),(side*.56,y,-.32),(.025,.025,.32),material='wood')
 for g in ET.parse(baby.XML).getroot().findall('.//geom'):
  if not(g.get('name')or'').startswith('doll_'):continue
  g=copy.deepcopy(g)
  g.set('size',' '.join(format(float(x)*2.5,'.12g')for x in g.get('size').split()))
  if g.get('fromto'):
   p=np.fromstring(g.get('fromto'),sep=' ').reshape(2,3)*2.5;p[:,2]-=.0275;g.set('fromto',' '.join(format(float(x),'.12g')for x in p.ravel()))
  else:
   p=np.fromstring(g.get('pos','0 0 0'),sep=' ')*2.5;p[2]-=.0275;g.set('pos',' '.join(format(float(x),'.12g')for x in p))
  body.append(g)
 for name,pos in [('bottom_site','0 0 -.64'),('top_site','0 0 .45'),('horizontal_radius_site','.70 0 0')]:ET.SubElement(wrap,'site',name=name,pos=pos,size='.001')
 ET.indent(root);ET.ElementTree(root).write(XML,encoding='unicode')

def configuration():
 c=copy.deepcopy(parent.configuration());c['crib'].update(xml=str(XML),initial_z_m=.64,footprint_half_xy_m=[.632,.332],inside_half_xy_m=[.60,.30],inside_bottom_z_m=.040,inside_top_z_m=.45)
 c['layout']['crib_dy_m']=0.;return c

class Scene(parent.T1R4CableScene):
 def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
  original=parent.configuration;sample=tabletop.sample_slot
  def config():
   with patch.object(parent,'configuration',original):return configuration()
  def own_slot(slot,seed,object_index=0):
   xyz,yaw=sample(slot,seed,object_index=object_index)
   if slot=='R':xyz=xyz+np.array([.16,.66,-.26])
   return xyz,yaw
  with patch.object(parent,'configuration',config),patch.object(tabletop,'sample_slot',own_slot):super().__init__(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
  self.source_scene_xml=self.backend.model.get_xml()
