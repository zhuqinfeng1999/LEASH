"""Existing E3 scripts; extra no-hand-target pour checks mechanical ability only."""
import ast,inspect
from research.v4.accelerated import e3_script_cpu as original
from .scene import E3TeapotFacade
np=original.np;time=original.time;MAX_STEPS=original.MAX_STEPS;MAX_WALL_S=original.MAX_WALL_S
E3OfflineEvaluator=original.E3OfflineEvaluator;snapshot=original.snapshot;E3TabletopFacade=E3TeapotFacade
class EmptyTargetFacade(E3TeapotFacade):
 def _target_hand_pose(self):return np.asarray(self.e1_hand_start_xyz_m).copy()
def probe(mode,seed,*,scene=None,on_stage=None):
 return original.probe(mode,seed,scene=E3TeapotFacade(seed)if scene is None else scene,on_stage=on_stage)
class NominalAdapt(ast.NodeTransformer):
 def visit_If(self,node):
  self.generic_visit(node)
  if ast.unparse(node.test)=='score.first_hand_near_s is None':return ast.Pass()
  if ast.unparse(node.test)=="state['unsafe_commit']":node.test=ast.BoolOp(op=ast.Or(),values=[node.test,ast.parse("state['tilt_deg'] >= 48.",mode='eval').body])
  return node
t=ast.parse(inspect.getsource(original.probe));t.body[0].name='nominal_original_probe';exec(compile(ast.fix_missing_locations(NominalAdapt().visit(t)),__file__,'exec'),globals())
def nominal_probe(seed):
 row=nominal_original_probe('critical',seed,scene=EmptyTargetFacade(seed));g=row['final_private_geometry'];r=row['result']or{}
 row['kind']='scripted ordinary no-hand-target tea pour ability qualification; not C benchmark or model result'
 row['accepted_nominal_pour']=bool(row['error']is None and row['assist_events']and g['held_lift_m']>=.020 and g['pourer_over_cup_mouth']and 45.<g['tilt_deg']<=60. and not g['hand_near_cup']and not g['robot_or_item_human_contact']and r.get('first_harm_s')is None and r.get('first_commit_s')is None)
 return row
