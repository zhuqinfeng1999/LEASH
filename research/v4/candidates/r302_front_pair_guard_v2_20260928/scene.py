import hashlib
import json
from pathlib import Path
from types import FunctionType
import mujoco
from research.v4.candidates.r302_front_pair_contact_20260928 import scene as pair
from .guard import compare
from .admit import verify
H=Path(__file__).resolve().parent
_RAISED=FunctionType(pair._RAISED.__code__,dict(pair._RAISED.__globals__,H=H),closure=pair._RAISED.__closure__)
_FRONT=FunctionType(pair._ORIGINAL_FRONT.__code__,dict(pair._ORIGINAL_FRONT.__globals__,H=H,_RAISED_INIT=_RAISED),closure=pair._ORIGINAL_FRONT.__closure__)
def selected_front_init(self,*args,**kwargs):
    _FRONT(self,*args,**kwargs)
    m=self.backend.sim.model._model
    previous=mujoco.MjModel.from_binary_path(str(pair.old.H/'startup_seed0/compiled_model.mjb'))
    # Save the actual new model even if metadata guard fails; no re-constructor.
    target=H/'startup_seed0/compiled_model.mjb';mujoco.mj_saveModel(m,str(target))
    r=compare(previous,m,pair.PARAMS);r['before_any_startup_controls']=True
    r['actual_compiled_model_before_startup_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    (H/'compiled_pair_guard.json').write_text(json.dumps(r,indent=2)+'\n')
    if not r['passed']:raise RuntimeError('First unexplained pair guard-v2 difference: '+repr(r['failures']))
_INIT=FunctionType(pair._INIT.__code__,dict(pair._INIT.__globals__,H=H,_FRONT_INIT=selected_front_init,
        sample=FunctionType(pair.old.sample.__code__,dict(pair.old.sample.__globals__,H=H)),verify=verify),closure=pair._INIT.__closure__)
class SourcePairGuardV2CanScene(pair.SourcePairFixed80CanScene):
    __init__=_INIT
    layout_version='R302-F-shortV-fixed80-two-support20ms-pairs-exact-name-allocation-guard-v2'
