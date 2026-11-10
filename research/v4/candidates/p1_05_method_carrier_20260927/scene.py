"""Method-only carrier; accepted common MULTICCD16 physics unchanged.

Qualification timing/rows belong to the offline script, not the method.
The method runner owns its strict evaluator and48/144/80/1800 limits.
"""
from research.v4.candidates.p1_05_multiccd_qualification_20260927.scene import Scene as QualificationScene
from research.v4.candidates.p1_05_exact_clearance_v3_20260926.scene import P1DumbbellFacade as PhysicalScene
from research.v4.candidates.p1_05_exact_clearance_v3_20260926.scorer import P1BasketEvaluator

class Scene(QualificationScene):
 method_carrier_revision='P105_COMMON_MULTICCD16_METHOD_NO_OFFLINE_DIAGNOSTIC_CAPS_v1'
 def __init__(self,*args,**kwargs):
  # Exact accepted constructor enables flag16 and verifies every model array
  # and all other options. No replacement initialization/control/assist.
  super().__init__(*args,**kwargs)
  self.common_option_changes=dict(self.case_diag['option_changes'])
  # These are exclusively offline logging/cache fields, not Mujoco state,
  # controller state, assist, dynamics or task event fields.
  del self.case_diag,self._case_started,self._private_evaluator
 def step(self,action):
  # Same effective physical method called by QualificationScene.super.step.
  # Keep actual assist/hand/person/backend/raw update in its original MRO.
  return PhysicalScene.step(self,action)
 def close(self):
  # Offline private final_score would have an unobserved cache. The method
  # runner must use its own unchanged P1BasketEvaluator per actual step.
  return PhysicalScene.close(self)
