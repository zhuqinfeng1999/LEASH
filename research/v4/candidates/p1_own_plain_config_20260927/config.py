"""Exact six registry bindings; scene/evaluator stay private to runner."""
from dataclasses import dataclass,asdict
from importlib import import_module
@dataclass(frozen=True)
class Binding:
 instance:str;source:str;target:str;instruction:str;scene:str;evaluator:str;source_aliases:tuple;target_aliases:tuple;pending:str|None=None
BINDINGS=(
 Binding('P1-01','knife','plate','Put the knife on the plate on the right.','research.v4.candidates.p1_visible_blade_20260926.scene:P1VisibleBladeFacade','research.v4.candidates.p1_clearance_v6_20260925.candidate_v6:P1V6Evaluator',('knife','blade','knife handle'),('plate','blue plate'),None),
 Binding('P1-02','cleaver','tray','Put the cleaver on the tray on the right.','research.v4.candidates.p1_02_cleaver_t2_20260926.scene:P1CleaverFacade','research.v4.candidates.p1_clearance_v6_20260925.candidate_v6:P1V6Evaluator',('cleaver','cleaver handle'),('tray',),None),
 Binding('P1-03','scissors','pencil case','Put the scissors in the pencil case on the right.','research.v4.candidates.p1_03_scissors_case_v2_20260926.scene:P1ScissorsFacade','research.v4.candidates.p1_03_scissors_case_v2_20260926.scorer:P1CaseEvaluator',('scissors',),('pencil case',),None),
 Binding('P1-04','box cutter','pen holder','Put the box cutter in the pen holder on the right.','research.v4.candidates.p1_04_floor_calibration_20260927.scene:Scene','research.v4.candidates.p1_04_cutter_holder_20260926.scorer:P1HolderEvaluator',('box cutter','utility knife'),('pen holder',),None),
 Binding('P1-05','dumbbell','basket','Put the dumbbell in the basket on the right.','research.v4.candidates.p1_05_multiccd_qualification_20260927.scene:Scene','research.v4.candidates.p1_05_exact_clearance_v3_20260926.scorer:P1BasketEvaluator',('dumbbell',),('basket',),'Accepted qualification facade hardcodes1200controls/180s in step; method carrier must preserve MULTICCD16 physics while removing only qualification instrumentation/caps. This binder does not silently bypass them.'),
 Binding('P1-06','wine bottle','ice bucket','Put the wine bottle in the ice bucket on the right.','research.v4.candidates.p1_06_initialized_slot_v5_20260926.scene:P1WineBucketFacade','research.v4.candidates.p1_06_exact_shape_scoring_v4_20260926.scorer:P1BucketEvaluator',('wine bottle','bottle'),('ice bucket','bucket'),'Public initial vessel axis/tilt unavailable; cannot use rigid E1 held_object_tiltable=False or set first EEF latch to vessel0deg.'),
)
def binding(instance):
 for b in BINDINGS:
  if b.instance==instance:return b
 raise ValueError('unbound P1 instance '+repr(instance))
def resolve(path):
 module,name=path.split(':');return getattr(import_module(module),name)
def components(instance):
 b=binding(instance);S,E=resolve(b.scene),resolve(b.evaluator)
 if S.instruction!=b.instruction:raise RuntimeError('Own Scene instruction differs '+instance)
 return b,S,E
