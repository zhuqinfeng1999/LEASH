"""One authorized matching whole-doll visual/collision3mm layout correction."""
import copy,xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.candidates.r4_crib_doll_collision_20260926 import candidate as parent
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];XML=HERE/'crib_doll_raised3mm.xml';SHIFT=np.array([0.,0.,.003])
def generate():
 root=ET.parse(parent.XML).getroot();changed=[]
 for g in root.findall('.//geom'):
  if not(g.get('name')or'').startswith('doll_'):continue
  if g.get('fromto'):
   p=np.fromstring(g.get('fromto'),sep=' ').reshape(2,3)+SHIFT;g.set('fromto',' '.join(format(float(x),'.12g')for x in p.ravel()))
  else:p=np.fromstring(g.get('pos','0 0 0'),sep=' ')+SHIFT;g.set('pos',' '.join(format(float(x),'.12g')for x in p))
  changed.append(g.get('name'))
 if len(changed)!=38:raise RuntimeError('Full doll union size changed')
 ET.indent(root);ET.ElementTree(root).write(XML,encoding='unicode');return changed
def scene_factory(instance='R4-06'):
 original=parent.scene_factory(instance)
 class RaisedDollScene(original):
  def __init__(self,*args,**kwargs):
   with patch.object(parent,'XML',XML):super().__init__(*args,**kwargs)
 RaisedDollScene.__name__=instance.replace('-','')+'DollRaised3mm';return RaisedDollScene
