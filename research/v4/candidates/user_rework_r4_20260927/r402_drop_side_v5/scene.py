"""Normal full-size crib with one visible lowered center access panel."""
import copy,json,xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch
from ..r402_robot_side_v3 import scene as carrier
from ..r402_lowered_rail_v2 import scene as old_lowered
from ..r402_v1 import scene as full_height
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
XML=HERE/'floor_crib_drop_side.xml'

def generate():
 root=ET.parse(full_height.XML).getroot();body=root.find('.//body[@name="object"]')
 if body is None:raise ValueError('Exact original furniture body missing')
 changes=[]
 for g in list(body.findall('geom')):
  name=g.get('name','')
  if name.startswith('front_rail_'):
   oldpos=[float(x)for x in g.get('pos').split()];oldsize=[float(x)for x in g.get('size').split()]
   if oldsize!=[.62,.012,.015]:raise ValueError('Exact full-height rail changed')
   body.remove(g)
   for side in(-1,1):
    q=copy.deepcopy(g);q.set('name',name+('_left_fixed'if side<0 else'_right_fixed'))
    q.set('pos',f'{side*.46:.12g} {oldpos[1]:.12g} {oldpos[2]:.12g}');q.set('size','.16 .012 .015');body.append(q)
   q=copy.deepcopy(g);q.set('name',name+'_center_lowered');q.set('pos',f'0 {oldpos[1]:.12g} {oldpos[2]-.5:.12g}');q.set('size','.30 .012 .015');body.append(q)
   changes.append(dict(name=name,original_pos=oldpos,segment_half_width_m=.30,translation_z_m=-.5))
  elif name.startswith('front_slat_'):
   pos=[float(x)for x in g.get('pos').split()]
   if abs(pos[0])<=.300000001:
    original_pos=pos[:];pos[2]-=.5;g.set('pos',' '.join(format(x,'.12g')for x in pos));g.set('name',name+'_center_lowered');changes.append(dict(name=name,original_pos=original_pos,translation_z_m=-.5))
 # Visible wood guides are the physical sliding supports at the panel edges.
 for x in(-.316,.316):
  for visual in(False,True):
   ET.SubElement(body,'geom',name=f'front_access_slide_guide_{x}'+('_visual'if visual else'_collision'),type='box',pos=f'{x} -.335 .2525',size='.012 .014 .1975',group='1'if visual else'0',contype='0'if visual else'1',conaffinity='0'if visual else'1',mass='0',material='wood',friction='1 .005 .001',solref='.02 1')
 root.set('model','original_normal_crib_center_drop_side_v5');ET.indent(root);ET.ElementTree(root).write(XML,encoding='unicode')
 (HERE/'geometry_changes.json').write_text(json.dumps(dict(original_source=str(full_height.XML.relative_to(ROOT)),changes=changes,whole_crib_or_infant_scale_changed=False,matress_top_m=.68,lowered_center_highest_rail_world_z_m=.64+.435-.5+.015,other_sides_full_height=True,risk_or_assist_changed=False),indent=2)+'\n')

class Scene(carrier.Scene):
 def __init__(self,*a,**kw):
  with patch.object(old_lowered,'XML',XML):super().__init__(*a,**kw)
  self.furniture_access_version='normal-fullsize-center-drop-side-v5'

if __name__=='__main__':generate()
