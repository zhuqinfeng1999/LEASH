"""One normal20mm higher seat; complete human translated, unchanged arm shape."""
import xml.etree.ElementTree as ET
import numpy as np
from unittest.mock import patch
from . import scene as original
from research.v4.candidates.user_rework_p2_20260927.scene_v3 import NaturalReceivingPerson
CONFIG=original.CONFIG;PREFIX=original.PREFIX;PAN_NAME=original.PAN_NAME
SHIFT=np.array([0.,0.,.020])
LEFT_HAND=tuple(np.array(original.LEFT_HAND)+SHIFT)
RIGHT_HAND=tuple(np.array([.125,.32,.938])+SHIFT)
instructions=original.instructions

class RaisedReceivingPerson(NaturalReceivingPerson):
 def pose(self,hand,right_hand=None):
  right=self.right_hand if right_hand is None else right_hand
  if right is None:raise ValueError('Exact complete receiving person requires right endpoint')
  posed=super().pose(np.asarray(hand)-SHIFT,np.asarray(right)-SHIFT)
  return {n:(r,t+SHIFT)for n,(r,t)in posed.items()}
 def augment(self,xml,hand):
  result=super().augment(xml,hand);root=ET.fromstring(result);world=root.find('worldbody')
  for name in('person_chair','person_footrest'):
   matches=[b for b in world.findall('body')if b.get('name')==name]
   if len(matches)!=1:raise ValueError('Exact chair/footrest identity missing')
   body=matches[0];pos=np.fromstring(body.get('pos'),sep=' ');body.set('pos',' '.join(map(str,pos+SHIFT)))
   # Normal taller support legs retain their original ground contact while the
   # seat/footrest and person rise20mm. No humanoid/scene collider is altered.
   legs=[g for g in body.findall('geom')if g.get('name')is None]
   if len(legs)!=(4 if name=='person_chair'else2):raise ValueError('Exact original support legs missing')
   for g in legs:
    if g.get('contype')!='0'or g.get('conaffinity')!='0':raise ValueError('Leg unexpectedly participates in physics')
    p=np.fromstring(g.get('pos'),sep=' ');s=np.fromstring(g.get('size'),sep=' ');p[2]-=.010;s[-1]+=.010
    g.set('pos',' '.join(map(str,p)));g.set('size',' '.join(map(str,s)))
  return ET.tostring(root,encoding='unicode')

def factory(instance):
 own=original.factory(instance)
 class HigherSeatOwnFacade(own):
  def __init__(self,*a,**kw):
   with patch.object(original,'NaturalReceivingPerson',RaisedReceivingPerson),patch.object(original,'LEFT_HAND',LEFT_HAND),patch.object(original,'RIGHT_HAND',RIGHT_HAND):super().__init__(*a,**kw)
 HigherSeatOwnFacade.__name__=instance.replace('-','')+'RaisedReceivingFacade'
 return HigherSeatOwnFacade

def create(seed,*,instance='S1-01',camera_observations=False,render_gpu_device_id=-1):
 f=factory(instance)(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
 return f.backend,f.grasp_assist
