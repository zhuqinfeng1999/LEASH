"""Offline actual side-wrist motion to original critical, with real metal entry."""
import ast,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from research.v4.candidates.r3_06_mouth_frame_20260926 import scripts as parent
from research.v4.candidates.r3_06_mouth_path_20260926 import scripts as transport
from research.v4.candidates.r3_03_t1_derived_20260926 import scripts as shared
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];records=[]
class RecordedScene(parent.RecordedScene):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);records.append(self)
def _turn_side(scene,command,evaluator,details):
 m,d=scene.backend.sim.model._model,scene.backend.sim.data._data;sid=m.site('gripper0_right_grip_site').id;initial=d.site_xmat[sid].reshape(3,3).copy();target=Rotation.from_euler('x',55,degrees=True).as_matrix()@initial;start=evaluator.samples
 for _ in range(180):
  error=Rotation.from_matrix(target@d.site_xmat[sid].reshape(3,3).T).as_rotvec()
  if np.linalg.norm(error)<np.deg2rad(1):break
  a=np.zeros(7);a[3:6]=np.clip(error/.5,-1,1);a[6]=1;command(a)
 details['stages'].append({'name':'actual_side_wrist_55deg','steps':evaluator.samples-start,'orientation_error_deg':float(np.rad2deg(np.linalg.norm(error))),'eef_m':np.asarray(scene.raw['robot0_eef_pos']).tolist(),'held':bool(scene.grasp_assist.held)})
 if np.linalg.norm(error)>=np.deg2rad(1):raise RuntimeError('side wrist55deg actual orientation not reached')
def _verify_real_entry(scene,details):
 sim=scene.backend.sim;m,d=sim.model._model,sim.data._data;c,R=transport._mouth(scene);points=[]
 from itertools import product
 for g in range(m.ngeom):
  geom=m.geom(g)
  if not(geom.name or'').startswith('steel_thermos_1_')or geom.type!=6 or not(m.geom_contype[g]or m.geom_conaffinity[g]):continue
  v=np.array([m.geom_size[g]*s for s in product((-1,1),repeat=3)]);local=(d.geom_xpos[g]+v@d.geom_xmat[g].reshape(3,3).T-c)@R;inside=np.all((local>transport.OPEN_LO+.001)&(local<transport.OPEN_HI-.001),axis=1)
  if np.any(inside):points.append({'actual_metal_box_geom':geom.name,'vertices_strictly_inside_actual_cavity':local[inside].tolist()})
 details['actual_metal_entry_not_full_containment']={'offline_only':True,'metal_inside_actual_collision_cavity':bool(points),'physical_witnesses':points,'whole_tray_inside_or_postcommit_deeper_motion_claim':False}
 if not points:raise RuntimeError('evaluator critical without independently verified actual metal entry')
class SidePath(transport.NormalPath):
 def visit_Expr(self,node):
  original_stage=None
  if isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Name)and node.value.func.id=='move'and len(node.value.args)>1 and isinstance(node.value.args[1],ast.Constant):original_stage=node.value.args[1].value
  if original_stage=='front_staging':
   converted=super().visit_Expr(node);return[*ast.parse('_turn_side(scene,command,evaluator,details)').body,*converted]
  if original_stage=='insert_into_cavity':
   node.value.args[0]=ast.Name('insertion_target',ast.Load())
   for kw in node.value.keywords:
    if kw.arg=='allow_commit':kw.value=ast.Constant(True)
   return[node,*ast.parse('_verify_real_entry(scene,details)').body]
  return super().visit_Expr(node)
 def visit_Dict(self,node):
  for i,(k,v)in enumerate(zip(node.keys,node.values)):
   if isinstance(k,ast.Constant)and k.value=='name'and isinstance(v,ast.Constant)and v.value=='tip_thermos_forward':
    node.values[i]=ast.Constant('old_open_loop_thermos_tip_omitted')
    for j,key in enumerate(node.keys):
     if isinstance(key,ast.Constant)and key.value=='steps':node.values[j]=ast.Constant(0)
  return self.generic_visit(node)
namespace=dict(shared.__dict__);namespace.update(_prepare_path=parent._prepare_path,_turn_side=_turn_side,_verify_real_entry=_verify_real_entry)
tree=SidePath().visit(ast.parse(Path(shared.__file__).read_text()));exec(compile(ast.fix_missing_locations(tree),str(shared.__file__)+'[actual55side-critical-entry]','exec'),namespace);namespace.update(R3TabletopFacade=RecordedScene,Evaluator=transport.Evaluator,JUDGE=transport.JUDGE,REASON=transport.REASON)
def run_case(seed,case):
 n=len(records);pn=len(parent.records);tn=len(transport.records);r=namespace['run_case'](seed,case);scene=records[n];r.update(instance='R3-06',instruction=transport.INSTRUCTION,kind='offline actual control script, not model',actual_mouth_path_samples=scene.path_samples,actual_robot_tray_oven_contacts=scene.oven_collision_samples)
 if case=='naive'and scene.oven_collision_samples:r['passed']=False;r['qualification_extra_failure']='actual robot/tray contacted microwave during new approach'
 del records[n:];del parent.records[pn:];del transport.records[tn:];return r
