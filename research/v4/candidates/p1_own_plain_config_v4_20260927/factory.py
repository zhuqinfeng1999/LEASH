"""P105 carrier opt-in only; all other v2 bindings preserved."""
from dataclasses import replace
from types import SimpleNamespace
from functools import lru_cache
from research.v4.candidates.p1_own_plain_config_v2_20260927 import factory as parent
from research.v4.candidates.p1_05_method_carrier_20260927.scene import Scene,P1BasketEvaluator
@lru_cache(maxsize=6)
def components(instance):
 c=parent.components(instance)
 if instance!='P1-05':return c
 assert c.Evaluator is P1BasketEvaluator,'Own strict evaluator must remain installed'
 b=replace(c.binding,scene='research.v4.candidates.p1_05_method_carrier_20260927.scene:Scene',pending=None)
 return SimpleNamespace(**{**vars(c),'binding':b,'Scene':Scene,'pending_reason':None,'cpu_call_path_ready':True})
BINDINGS=tuple(components(b.instance).binding for b in parent.BINDINGS)
