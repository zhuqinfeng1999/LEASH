"""One real forward source plinth; front vanity and original tool stay unchanged."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from research.v4.candidates.user_rework_p4_20260927.p402_front_vanity_v1 import scene as old
HERE=Path(__file__).resolve().parent
class ClearSourceEnv(old.FrontVanityEnv):
 def _load_model(self):
  super()._load_model();tray=self.model.worldbody.find("body[@name='heat_resistant_left_tray']");board=self.model.worldbody.find("body[@name='tabletop_ironing_board']")
  br=Rotation.from_quat(np.roll(np.fromstring(board.get('quat'),sep=' '),-1)).as_matrix();p=np.fromstring(tray.get('pos'),sep=' ')+br[:,1]*.13;tray.set('pos',' '.join(map(str,p)))
  for geom in list(tray.findall('geom')):tray.remove(geom)
  base=tray.find(".//geom[@name='p402_ceramic_base']");base.set('size','.14 .05 .016');p=np.fromstring(base.get('pos'),sep=' ');p[2]=.016;base.set('pos',' '.join(map(str,p)))
class Scene(old.Scene):
 layout_version='p402-own-front-vanity-source-forward130mm-real32mm-ceramic-v3'
 def __init__(self,*args,**kwargs):
  with patch.object(old,'FrontVanityEnv',ClearSourceEnv):super().__init__(*args,**kwargs)
