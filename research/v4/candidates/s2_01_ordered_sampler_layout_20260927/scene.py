"""Approved ONE actual-center layout + truthful sampler radii, no solid changes."""
import hashlib,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.candidates.s2_01_release_props_20260927 import scene as props
from research.v4.candidates.s2_public_reset_20260926 import scene as parked
HERE=Path(__file__).resolve().parent
ORIGINAL_REGISTER=props.register;ORIGINAL_FACTORY=parked.parked_env_factory
RADII={'milk_mug':.086,'pill_dish':.121,'basket':.104}
INIT_ATTEMPTS=[]
def rng():
 s=np.random.get_state();return hashlib.sha256(s[0].encode()+s[1].tobytes()+str(s[2:]).encode()).hexdigest()
def register():
 ORIGINAL_REGISTER()
 from libero.libero.envs.base_object import OBJECTS_DICT
 for category,radius in RADII.items():
  original=OBJECTS_DICT[category]
  def factory(*args,_original=original,_radius=radius,**kwargs):
   obj=_original(*args,**kwargs)
   site=obj.worldbody.find("./body/site[@name='"+obj.naming_prefix+"horizontal_radius_site']")
   if site is None:raise ValueError('original radius metadata site missing '+obj.name)
   p=np.fromstring(site.get('pos'),sep=' ');p[0]=_radius;site.set('pos',' '.join(map(str,p)))
   if not np.isclose(obj.horizontal_radius,_radius):raise ValueError('radius metadata alias failed')
   return obj
  OBJECTS_DICT[category]=factory
def layout_factory(**options):
 options['bddl_file_name']=str(HERE/'scene.bddl');return ORIGINAL_FACTORY(**options)
class Scene(props.Scene):
 protocol='S201-CC0-props+preseed+ordered20mm-centers+truthfulsampler-radii-v1'
 def __init__(self,seed=0,*args,**kwargs):
  record={'seed':int(seed),'before_preseed':rng()};np.random.seed(int(seed));record['after_preseed']=rng()
  try:
   with patch.object(props,'register',register),patch.object(parked,'parked_env_factory',layout_factory):
    super().__init__(seed=seed,*args,**kwargs)
  except Exception as exc:
   record.update(returned=False,error=f'{type(exc).__name__}: {exc}',after_constructor=rng());INIT_ATTEMPTS.append(record);raise
  record.update(returned=True,error=None,after_constructor=rng());self.initialization_rng_record=record;INIT_ATTEMPTS.append(record)
