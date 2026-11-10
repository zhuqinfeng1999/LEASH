"""Same normal floor crib, physically lowered near-side rail."""
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET
from ..r402_v1 import scene as parent
HERE=Path(__file__).resolve().parent;ROOT=parent.ROOT;XML=HERE/'floor_crib_lowered.xml'
def generate():
 root=ET.parse(parent.XML).getroot()
 for g in root.findall('.//geom'):
  name=g.get('name','')
  if name.startswith('front_rail_0.435'):
   g.set('pos','0 -.31 .075')
  elif name.startswith('front_slat_'):
   xyz=g.get('pos').split();xyz[2]='.045';g.set('pos',' '.join(xyz));g.set('size','.01 .01 .015')
 ET.indent(root);ET.ElementTree(root).write(XML,encoding='unicode')
class Scene(parent.Scene):
 def __init__(self,*args,**kwargs):
  with patch.object(parent,'XML',XML):super().__init__(*args,**kwargs)
