"""Same qualified P2 real-control script, endpoints, readonly geometry log."""
import inspect
from research.v4.candidates.p2_v42_t3_20260925 import scripted_safe_slot_v2 as base
from .scene import P2ScrewdriverFacade,INSTRUCTION
from .evaluator import P2ScrewdriverEvaluator
from .geometry import snapshot
for key,value in vars(base).items():
 if not key.startswith('__'):globals().setdefault(key,value)
_recorders=[]
class Recorder(P2ScrewdriverFacade):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.initial_private_geometry=snapshot(self);self.final_private_geometry=None;_recorders.append(self)
 def close(self):
  if self.final_private_geometry is None:self.final_private_geometry=snapshot(self)
  return super().close()
P2Facade=Recorder;P2Evaluator=P2ScrewdriverEvaluator
s=inspect.getsource(base.probe).replace('def probe(', 'def _original_probe(',1);exec(compile(s,__file__,'exec'),globals())
def probe(mode,seed):
 before=len(_recorders);r=_original_probe(mode,seed);scene=_recorders[before];r.update(instance='P2-03',instruction=INSTRUCTION,grasp_mode='assisted',initial_private_geometry=scene.initial_private_geometry,final_private_geometry=scene.final_private_geometry,tool_semantics='layflat ontable;handle nearer person than sharp tip;nohuman grip-force claim');del _recorders[before:];return r
