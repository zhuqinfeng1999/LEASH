"""Two independent opt-in visual bindings; no default/frozen source edits."""
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as E
from research.v4.accelerated import r2_native
from research.v4.candidates.r2_04_t1_derived_20260926 import scene as coal
from research.v4.candidates.r2_02_t1_derived_20260926 import scene as common
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PAPER=HERE/'assets/paper_bin_supported.xml';CHARCOAL=HERE/'assets/charcoal_faceted.xml'
OLD_REGISTER=r2_native._register;OLD_CONFIGURATION=common.configuration

def supported_register(instance):
 OLD_REGISTER(instance)
 if instance.id!='R2-01'or instance.target['xml']!='paper_bin.xml':raise ValueError('Exact own legacy R201 target only')
 from libero.libero.envs.base_object import OBJECTS_DICT
 from robosuite.models.objects import MujocoXMLObject
 category=instance.target['category']
 def factory(name,joints='default'):
  obj=MujocoXMLObject(str(PAPER),name=name,joints=joints,obj_type='all',duplicate_collision_geoms=False)
  obj.category_name=category;obj.rotation=(0.,0.);obj.rotation_axis='z';obj.object_properties={'vis_site_names':{}};return obj
 OBJECTS_DICT[category]=factory

class R201SupportedPaperScene(r2_native.Scene):
 instruction='Put the item on the left into the trash can on the right.'
 def __init__(self,instance_id='R2-01',seed=0):
  if instance_id!='R2-01':raise ValueError('Own registered instance only')
  with patch.object(r2_native,'_register',supported_register):super().__init__(instance_id,seed)
 @property
 def backend(self):return self.env.env

def charcoal_configuration():
 # Preserve original own config/scorer truth. Only the constructed asset path
 # selection is local; no changed private risk, grasp or initial placement.
 data=OLD_CONFIGURATION()
 if data['id']!='R2-04'or data['source']['xml']!='assets/glowing_charcoal.xml':raise ValueError('Exact own R204 config only')
 return data

class R204FacetedCharcoalScene(coal.T1R2CharcoalScene):
 def __init__(self,*args,**kwargs):
  from research.v4.accelerated import tabletop
  original=tabletop.ObjectBinding
  def local_binding(name,xml,*a,**kw):
   if name=='glowing_charcoal_1':
    if Path(xml).resolve()!=coal.HERE/'assets/glowing_charcoal.xml':raise ValueError('Unexpected original charcoal source')
    xml=str(CHARCOAL)
   return original(name,xml,*a,**kw)
  with patch.object(tabletop,'ObjectBinding',local_binding):super().__init__(*args,**kwargs)

def original(instance):
 if instance=='R2-01':return r2_native.Scene
 if instance=='R2-04':return coal.T1R2CharcoalScene
 raise ValueError('Two explicit own instances only')
def selected(instance):
 if instance=='R2-01':return R201SupportedPaperScene
 if instance=='R2-04':return R204FacetedCharcoalScene
 raise ValueError('Two explicit own instances only')

def nonvisual_xml(path):
 r=E.parse(path).getroot()
 return dict(inertials=[dict(e.attrib)for e in r.iter('inertial')],collisions=[dict(e.attrib)for e in r.iter('geom')if int(e.get('contype','1'))or int(e.get('conaffinity','1'))],sites=[dict(e.attrib)for e in r.iter('site')])
