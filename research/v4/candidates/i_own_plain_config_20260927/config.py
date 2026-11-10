"""Own accepted T7 constructors and strict private scorers, no scene text online."""
from dataclasses import dataclass
from importlib import import_module
from types import SimpleNamespace
from research.v4.candidates.i1_v42_t1t7_20260925.facade import I1Facade as PublicFacade
from research.v4.candidates.i_t7_physical_support_v4_20260926.evaluator import Evaluator as StrictEvaluator
@dataclass(frozen=True)
class Binding:
 instance:str;source:str;target:str;other:str;instruction:str;create_module:str
 pending:str='Own native public masks/held-axis and landing certificate have not been validated for this ordinary-tool protocol; no method launch authorized.'
INSTRUCTION='Pick up the ketchup and place it in the basket.'
BINDINGS=tuple(Binding(f'{family}-0{n}','ketchup','basket','candle'if n==2 else 'metal can',INSTRUCTION,
 'research.v4.candidates.i4_named_label_pose_v2_20260926.scene'if family=='I4'and n==2 else 'research.v4.candidates.i4_can_label_pose_v3_20260926.scene'if family=='I4'else 'research.v4.candidates.i_t7_candle_can_derived_20260926.scene')for family in ('I1','I2','I3','I4')for n in (2,3))
def binding(instance):return next(b for b in BINDINGS if b.instance==instance)
def components(instance):
 b=binding(instance);create=import_module(b.create_module).create
 class Scene(PublicFacade):
  instruction=b.instruction
  def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
   self.backend,self.grasp_assist=create(instance,int(seed),camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
   self.env=SimpleNamespace(sim=self.backend.sim,env=self.backend);self.raw=self.backend._get_observations(force_update=True);self.sequence=0;self.assist_events=[]
 class Evaluator(StrictEvaluator):
  def __init__(self):super().__init__(instance)
 return b,Scene,Evaluator
