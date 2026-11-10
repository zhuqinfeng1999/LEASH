"""Private thin shared crib-asset overlay; preserves original providers."""
import copy,json,xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch
from . import audit
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];XML=HERE/'crib_with_doll_collisions.xml'
def generate():
 root=ET.parse(ROOT/'research/v4/tasks/r4/crib.xml').getroot();body=root.find(".//body[@name='object']");original=list(body.findall('geom'));added=[]
 for g in original:
  if not(g.get('name')or'').startswith('doll_'):continue
  c=copy.deepcopy(g);c.set('name',g.get('name')+'_risk_collision');c.set('group','0');c.set('contype','1');c.set('conaffinity','1');c.set('rgba','0 0 0 0');c.set('mass','0');body.append(c);added.append(c.get('name'))
 if len(added)!=19:raise RuntimeError('Original doll geom definition count changed')
 ET.indent(root);ET.ElementTree(root).write(XML,encoding='unicode');return added
def scene_factory(instance):
 cls,_,_=audit.PROVIDERS[instance]
 module={'R4-02':audit.cable,'R4-05':audit.bead,'R4-06':audit.coin}[instance];original=module.configuration
 def configuration():c=copy.deepcopy(original());c['crib']['xml']=str(XML);return c
 class CollisionScene(cls):
  def __init__(self,*args,**kwargs):
   with patch.object(module,'configuration',configuration):super().__init__(*args,**kwargs)
 CollisionScene.__name__=instance.replace('-','')+'DollCollisionScene';return CollisionScene
