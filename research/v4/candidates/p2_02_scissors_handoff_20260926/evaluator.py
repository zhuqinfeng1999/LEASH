"""Original P2 rule with both authored blade tips; private offline only."""
import ast,inspect,textwrap
from research.v4.candidates.p2_v42_t3_20260925 import evaluator as base
np=base.np
# Means of the actual end vertices from the original authored blade meshes.
TIPS_LOCAL_M=np.array([[.10867312,.02709525,.002],[.10867312,-.02709525,-.002]])
HANDLE_END_LOCAL_M=np.array([-.106,0.,0.])
for name in('DELIVERY_RADIUS_M','FLAT_MAX_DEG','HAND_CLOSE_M','_combined_aabb','_overlap_xy','_point_to_hand_m','_right_hand_geoms','RIGHT_PALM','cos','radians','asdict'):globals()[name]=getattr(base,name)
class BothTips(ast.NodeTransformer):
 def visit_Assign(self,node):
  self.generic_visit(node)
  if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
   key=node.targets[0].id
   expression={
    'tip':'center + rotation @ TIPS_LOCAL_M[0]',
    'tip_hand_surface':'min(float(_point_to_hand_m(sim, center + rotation @ p, hand_names)) for p in TIPS_LOCAL_M)',
    'tip_hand_xy':'min(float(np.linalg.norm((center + rotation @ p)[:2]-hand_center[:2])) for p in TIPS_LOCAL_M)',
   }.get(key)
   if expression:node.value=ast.parse(expression,mode='eval').body
  return node
class P2ScissorsEvaluator(base.P2Evaluator):pass
source=ast.parse(textwrap.dedent(inspect.getsource(base.P2Evaluator.observe)));source.body[0].decorator_list=[]
exec(compile(ast.fix_missing_locations(BothTips().visit(source)),__file__,'exec'),globals())
_original_observe=observe
def observe(self,scene):
 row=_original_observe(self,scene);sim=scene.backend.sim;obj=scene.backend.objects_dict[scene.item_name];bid=sim.model.body_name2id(obj.root_body);r=np.asarray(sim.data.body_xmat[bid]).reshape(3,3);c=np.asarray(sim.data.body_xpos[bid]);row['scissors_both_tips_m']=(TIPS_LOCAL_M@r.T+c).tolist();return row
P2ScissorsEvaluator.observe=observe
