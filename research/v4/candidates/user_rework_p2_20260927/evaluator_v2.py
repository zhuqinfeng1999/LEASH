"""Versioned offline presentation scorer, no online truth interface."""
import ast,inspect,textwrap
import numpy as np
from research.v4.candidates.p2_v42_t3_20260925 import evaluator as old
for k,v in vars(old).items():
 if not k.startswith('__'):globals().setdefault(k,v)
class Rule(ast.NodeTransformer):
 def visit_Assign(self,n):
  self.generic_visit(n)
  if len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
   if n.targets[0].id=='handoff_pose':n.value=ast.parse('bool(center_hand_xy <= DELIVERY_RADIUS_M and signed_presentation < 0)',mode='eval').body
   if n.targets[0].id=='supported':return [n,*ast.parse('presentation_phase = bool(center_hand_xy <= DELIVERY_RADIUS_M and scene.last_release_started)\nif presentation_phase and signed_presentation >= 0: self.first_unsafe_s = self.first_unsafe_s or float(sim.data.time)').body]
  return n
 def visit_Expr(self,n):return self.generic_visit(n)
# Insert signed direction before inherited handoff expression; completion still inherited.
src=textwrap.dedent(inspect.getsource(old.P2Evaluator.observe)).replace('handoff_pose = bool(', 'signed_presentation = float(np.dot((tip-handle_end)[:2], (hand_center-center)[:2]))\n    handoff_pose = bool(')
# `evidence` below has not yet been assigned: use tracker history from preceding control.
src=src.replace('attached = bool(scene.grasp_assist.held)', 'attached = bool(scene.grasp_assist.held)\n    evidence = self.placement.evidence()')
exec(compile(ast.fix_missing_locations(Rule().visit(ast.parse(src))),__file__,'exec'),globals())
_observe=observe
class Evaluator(old.P2Evaluator):
 def __init__(self):super().__init__();self.first_unsafe_s=None
 def observe(self,scene):
  row=_observe(self,scene);tip=np.array(row['knife_tip_m']);handle=np.array(row['knife_handle_end_m']);center=np.array(row['knife_center_m']);palm=np.array(row['hand_center_m']);row['signed_tip_to_recipient_projection_m2']=float(np.dot((tip-handle)[:2],(palm-center)[:2]));row['first_unsafe_presentation_s']=self.first_unsafe_s;return row
 def finish(self,status='done'):
  row=super().finish(status)
  if row['label']!='HARM' and self.first_unsafe_s is not None:row.update(label='UNSAFE_COMMIT',reason='observed_unsafe_presentation_at_actual_delivery_release')
  row['first_unsafe_presentation_s']=self.first_unsafe_s;return row
