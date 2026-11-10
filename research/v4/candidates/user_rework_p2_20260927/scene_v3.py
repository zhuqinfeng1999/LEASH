"""Natural hanging left wrist; v1/v2 left-angle evidence preserved."""
from unittest.mock import patch
import numpy as np
from research.safevla.person import between
from . import scene as original
LEFT_HAND=original.LEFT_HAND;RIGHT_HAND=original.RIGHT_HAND
class NaturalReceivingPerson(original.ReceivingPerson):
 def pose(self,hand,right_hand=None):
  pose=super().pose(hand,right_hand)
  re,te=pose['lowerarm01.L'];elbow=re@self.head['lowerarm01.L']+te
  r,t=pose['wrist.L'];wrist=r@self.head['wrist.L']+t
  old_axis=r@(self.head['finger3-1.L']-self.head['wrist.L']);down_axis=wrist-elbow
  g=between(old_axis,down_axis)
  for name in self.names:
   cursor=name
   while cursor and cursor!='wrist.L':cursor=self.bones[cursor]['parent']
   if cursor:
    r,t=pose[name];pose[name]=(g@r,wrist+g@(t-wrist))
  return pose
class Scene(original.Scene):
 def __init__(self,*a,**kw):
  with patch.object(original,'ReceivingPerson',NaturalReceivingPerson):super().__init__(*a,**kw)
