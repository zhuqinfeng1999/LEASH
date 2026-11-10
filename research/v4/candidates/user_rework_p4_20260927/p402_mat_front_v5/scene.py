"""Exact approved one mat/rib translation; root-accepted source version unchanged."""
from unittest.mock import patch
import numpy as np
from research.v4.candidates.user_rework_p4_20260927.p402_clear_source_plinth_v3 import scene as old
from research.v4.candidates.user_rework_p4_20260927.p402_front_vanity_v1 import scene as front
SHIFT=.009989783829413654
class FrontMatEnv(old.ClearSourceEnv):
 def _load_model(self):
  super()._load_model();board=self.model.worldbody.find("body[@name='tabletop_ironing_board']")
  for g in board.findall('geom'):
   if g.get('name')=='silicone_heat_mat'or(g.get('name')or'').startswith('heat_mat_rib_visual_'):
    p=np.fromstring(g.get('pos'),sep=' ');p[1]+=SHIFT;g.set('pos',' '.join(map(str,p)))
class Scene(old.Scene):
 layout_version='p402-front-source-plinth-exact-mat-forward9_9898mm-v5'
 def __init__(self,*args,**kwargs):
  # Invoke original exact v3 constructor with its Env entry scoped once.
  with patch.object(old,'ClearSourceEnv',FrontMatEnv):super().__init__(*args,**kwargs)
