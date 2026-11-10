"""One accepted P106 strict-scorer alias correction; v1 preserved."""
from dataclasses import replace
from types import SimpleNamespace
from functools import lru_cache
from research.v4.candidates.p1_own_plain_config_20260927 import factory as old,config
from research.v4.candidates.p1_06_initialized_slot_v5_20260926.scorer import P1BucketEvaluator

@lru_cache(maxsize=6)
def components(instance):
 c=old.components(instance)
 if instance!='P1-06':return c
 b=replace(c.binding,evaluator='research.v4.candidates.p1_06_initialized_slot_v5_20260926.scorer:P1BucketEvaluator')
 return SimpleNamespace(**{**vars(c),'binding':b,'Evaluator':P1BucketEvaluator})
BINDINGS=tuple(components(b.instance).binding for b in config.BINDINGS)
