"""Registered own oven-front cameras; wrist and mechanics unchanged."""
from unittest.mock import patch
from research.v4.accelerated.tabletop import _look_at_quat
from research.v4.candidates.r3_06_own_midpoint_qualification_20260928 import scripts
from research.v4.candidates.r3_06_oven_only_side55_prefix_20260928 import scene as parent
PUBLIC_POS=[-.45,-.65,1.65];DISPLAY_POS=[-.6,-.8,1.7];TARGET=[.10,-.01,1.06]
class FrontCartEnv(parent.OwnCartEnv):
 def _load_model(self):
  super()._load_model()
  for name,pos in [('agentview',PUBLIC_POS),('research_spectator',DISPLAY_POS)]:
   c=self.model.worldbody.find(".//camera[@name='"+name+"']")
   if c is None:raise ValueError('Registered own camera missing '+name)
   c.set('pos',' '.join(map(str,pos)));c.set('quat',' '.join(map(str,_look_at_quat(pos,TARGET))))
class Scene(scripts.RecordedScene):
 public_camera_revision='R3-front-camera-640-workspace-and-offline-spectator-v1-20260928'
 def __init__(self,*a,**kw):
  with patch.object(parent,'OwnCartEnv',FrontCartEnv):super().__init__(*a,**kw)
