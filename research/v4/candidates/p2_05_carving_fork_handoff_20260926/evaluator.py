"""Original T3 predicate; every actual distal tip-face vertex is checked."""
import ast,inspect,textwrap,itertools
from research.v4.candidates.p2_v42_t3_20260925 import evaluator as base
for k,v in vars(base).items():
 if not k.startswith('__'):globals().setdefault(k,v)
# Exact authored prong_tip boxes: center(.174,±.015,.002), halfsize(.009,.001,.001).
# Check all eight distal-face vertices, not a fictitious knife endpoint.
TIPS_LOCAL_M=np.array([[.183,y+dy,.002+dz]for y in(-.015,.015)for dy,dz in itertools.product((-.001,.001),repeat=2)])
TIP_CENTERS_LOCAL_M=np.array([[.183,-.015,.002],[.183,.015,.002]])
HANDLE_END_LOCAL_M=np.array([-.1205,0.,0.])
class AllTips(ast.NodeTransformer):
 def visit_Assign(self,node):
  self.generic_visit(node)
  if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
   expr={'tip':'center + rotation @ TIP_CENTERS_LOCAL_M[0]',
    'tip_hand_surface':'min(float(_point_to_hand_m(sim, center + rotation @ p, hand_names)) for p in TIPS_LOCAL_M)',
    'tip_hand_xy':'min(float(np.linalg.norm((center + rotation @ p)[:2]-hand_center[:2])) for p in TIPS_LOCAL_M)'}.get(node.targets[0].id)
   if expr:node.value=ast.parse(expr,mode='eval').body
  return node
source=ast.parse(textwrap.dedent(inspect.getsource(base.P2Evaluator.observe)))
exec(compile(ast.fix_missing_locations(AllTips().visit(source)),__file__,'exec'),globals())
_original_observe=observe
def observe(self,scene):
 row=_original_observe(self,scene);sim=scene.backend.sim;obj=scene.backend.objects_dict[scene.item_name];bid=sim.model.body_name2id(obj.root_body);rotation=np.asarray(sim.data.body_xmat[bid]).reshape(3,3);center=np.asarray(sim.data.body_xpos[bid]);row['carving_fork_actual_tip_faces_m']=(TIPS_LOCAL_M@rotation.T+center).tolist();row['carving_fork_two_tip_centers_m']=(TIP_CENTERS_LOCAL_M@rotation.T+center).tolist();return row
class P2CarvingForkEvaluator(base.P2Evaluator):pass
P2CarvingForkEvaluator.observe=observe
