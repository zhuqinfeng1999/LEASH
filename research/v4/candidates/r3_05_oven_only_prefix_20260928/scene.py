"""Opt-in oven-only reset transform, old own source/board/PARK exactly reused."""
import inspect,textwrap,math
import numpy as np
from unittest.mock import patch
from research.v4.candidates.r3_05_fork_floor_v3_20260926 import scene as own
from research.v4.candidates.r3_v42_tabletop_20260925 import scene as carrier
from research.v4.candidates.r3_02_device_cart_relayout_20260928 import scene as cart
from .layout import TRANSLATION

ORIGINAL_LOAD_MODEL=carrier._R3FixtureEnv._load_model

def cart_loader():
 src=textwrap.dedent(inspect.getsource(cart.CartFixtureEnv._load_model))
 old="    geometry.CradleFixtureEnv._load_model(self)\n    p,_=tabletop.sample_slot('L',self.layout_seed,object_index=0)\n    p[:2]+=OFFSET;p[2]=.9\n    cradle=self.model.worldbody.find(\"body[@name='r302_real_wood_v_cradle']\")\n    if cradle is None:raise ValueError('Real cradle seam changed')\n    cradle.set('pos',' '.join(map(str,p)))"
 if src.count(old)!=1:raise ValueError('Existing cart only loader seam changed')
 src=src.replace(old,'    base_loader(self)')
 ns=dict(cart.CartFixtureEnv._load_model.__globals__);ns['base_loader']=ORIGINAL_LOAD_MODEL
 exec(compile(src,__file__+'::original-load-plus-exact-existing-cart','exec'),ns)
 return ns['_load_model']
class OwnCartEnv(carrier._R3FixtureEnv):
 _load_model=cart_loader()

class OvenOnlyPrefixScene(own.T1R3ForkBowlScene):
 layout_version='R305-old-source-board-PARK-only-full-device-front-cart-v1'
 def __init__(self,*args,**kwargs):
  with patch.object(carrier,'_R3FixtureEnv',OwnCartEnv):super().__init__(*args,**kwargs)
  # Existing own initial/reset protocol completed. Change only oven joint;
  # physics source/board/robot initial qpos and old PARK are never touched.
  sim=self.backend.sim;obj=self.backend.objects_dict[self.oven_name]
  q=np.asarray(sim.data.get_joint_qpos(obj.joints[0]),float).copy()
  q[:3]+=np.asarray(TRANSLATION)
  w,x,y,z=q[3:];c=math.cos(-math.pi/4);s=math.sin(-math.pi/4)
  q[3:]=[c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w]
  sim.data.set_joint_qpos(obj.joints[0],q)
  self._anchor_and_open_microwave() # Existing reset/anchor, no step/control.
  if kwargs.get('camera_observations',False):self.raw=self.backend._get_observations(force_update=True)
