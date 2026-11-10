"""One full-domain lateral towel/normal wooden pull-out leaf derived position."""
import json,math
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.candidates.user_rework_p4_20260927.p402_source_envelope_front_v8.derive import trig_extreme
from research.v4.candidates.user_rework_p4_20260927.p402_vanity_backshift_v11 import scene as previous
from robosuite.utils.mjcf_utils import new_geom
H=Path(__file__).resolve().parent;A=math.radians(10);BACK=previous.SHIFT
old=json.loads((H.parent/'p402_old_towel_support_readonly/report.json').read_text());bodyX=old['source_hand_exact_one_adverse_domain_state']['source_body_board_xyz'][0]
HMAX=max(g['maximum'][0]for g in old['gripper_extents'])-bodyX
BRANCHES=[dict(interval=[lo,hi],maximum=trig_extreme(HMAX,-.20,.07+.04*sign,lo,hi,False))for lo,hi,sign in[(-A,0,-1),(0,A,1)]]
SOURCE_HAND_MAX_X=max(x['maximum'][0]for x in BRANCHES)
WOOD_HALF_X=.125;CENTER=SOURCE_HAND_MAX_X+.005+WOOD_HALF_X;DELTA_X=CENTER+.17
PARTS=[dict(name='p402_towel_front_leaf',center=[CENTER,.149,.1425],half=[.125,.029,.0075]),dict(name='p402_towel_left_leg',center=[CENTER-.1175,.1665,.0675],half=[.0075,.0075,.0675]),dict(name='p402_towel_right_leg',center=[CENTER+.1175,.1665,.0675],half=[.0075,.0075,.0675])]
class LateralTowelEnv(previous.RearVanityEnv):
 def _load_model(self):
  super()._load_model();board=self.model.worldbody.find("body[@name='tabletop_ironing_board']")
  for g in board.findall('geom'):
   if(g.get('name')or'').startswith('towel_'):
    p=np.fromstring(g.get('pos'),sep=' ');p+=np.array([DELTA_X,BACK,0]);g.set('pos',' '.join(map(str,p)))
  for part in PARTS:
   board.append(new_geom(name=part['name'],type='box',size=part['half'],pos=part['center'],rgba=[.38,.21,.09,1.],group=1,friction=[.8,.005,.0001],contype=1,conaffinity=1))
class Scene(previous.Scene):
 layout_version='one-full-domain-lateral-towel-real-wood-front-pullout'
 def __init__(self,*a,**kw):
  with patch.object(previous,'RearVanityEnv',LateralTowelEnv):super().__init__(*a,**kw)
