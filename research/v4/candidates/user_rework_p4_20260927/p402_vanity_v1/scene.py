from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.candidates.p4_02_flat_source_ceramic_v6_20260927 import scene as parent
HERE=Path(__file__).resolve().parent
class VanityEnv(parent.FlatCeramicEnv):
 def _load_model(self):
  super()._load_model()
  for name,shift in [('heat_resistant_left_tray',(-.28,-.04,0.)),('tabletop_ironing_board',(-.02,-.12,0.))]:
   body=self.model.worldbody.find(f"body[@name='{name}']");p=np.fromstring(body.get('pos'),sep=' ')+shift;body.set('pos',' '.join(map(str,p)))
class Scene(parent.Scene):
 layout_version='p402-normal-compact-drawer-vanity-teal-silicone-near-towel-v1'
 def __init__(self,*args,**kwargs):
  with patch.multiple(parent,HERE=HERE,FlatCeramicEnv=VanityEnv):super().__init__(*args,**kwargs)
