"""One geometry-derived normal tray path; shared R3 executor/scorer unchanged."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
from research.v3.tasks.p2.evaluator import _combined_aabb
from research.v4.candidates.r3_06_original_aluminum_tray_20260926 import qualify_cpu as original
from research.v4.candidates.r3_03_t1_derived_20260926 import scripts as shared
from research.v4.candidates.r3_06_original_aluminum_tray_20260926.scene import AluminumTrayScene
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTRUCTION='Put the item on the left into the microwave and close the door to heat it up.'
JUDGE=original.JUDGE;REASON=original.REASON;Evaluator=original.bound_evaluator;R3TabletopFacade=AluminumTrayScene
# Existing microwave actual collision-inner faces, explicitly derived from the
# frozen model's left/right/floor/roof/back box geometry. Not evaluator bounds.
OPEN_LO=np.array([-.2583,-.1584,.0432]);OPEN_HI=np.array([.1143,.1449,.3024]);records=[]
def _mouth(scene):
 sim=scene.backend.sim;obj=sim.model.body_name2id(scene.backend.objects_dict[scene.oven_name].root_body);return np.asarray(sim.data.body_xpos[obj]),np.asarray(sim.data.body_xmat[obj]).reshape(3,3)
def _gripper(scene):
 m=scene.backend.sim.model
 return [m.geom_id2name(i)for i in range(m.ngeom)if(m.geom_id2name(i)or'').startswith('gripper0_')and(m.geom_contype[i]or m.geom_conaffinity[i])]
def _corners(box):return np.array([[x,y,z]for x in box[:,0]for y in box[:,1]for z in box[:,2]])
def _geometry(scene):
 sim=scene.backend.sim;item=scene.backend.objects_dict[scene.item_name];c,R=_mouth(scene);tray=np.asarray(_combined_aabb(sim,item.contact_geoms));gripper=np.asarray(_combined_aabb(sim,_gripper(scene)));tv=(_corners(tray)-c)@R;gv=(_corners(gripper)-c)@R
 return dict(tray_world_aabb=tray,gripper_world_aabb=gripper,tray_oven_aabb=np.array([tv.min(0),tv.max(0)]),gripper_oven_aabb=np.array([gv.min(0),gv.max(0)]),oven_center=c,oven_rotation=R)
def _prepare_path(scene,details):
 sim=scene.backend.sim;g=_geometry(scene);eef=np.asarray(scene.raw['robot0_eef_pos']);obj=scene.backend.objects_dict[scene.item_name];center=np.asarray(sim.data.body_xpos[sim.model.body_name2id(obj.root_body)]);offset=center-eef
 if abs(g['oven_rotation'][2,2]-1)>1e-5 or abs(g['oven_rotation'][0,1])>1e-5:raise RuntimeError('frozen oven frame no longer world-aligned; stop rather than invent conversion')
 combined=np.array([np.minimum(g['tray_world_aabb'][0],g['gripper_world_aabb'][0]),np.maximum(g['tray_world_aabb'][1],g['gripper_world_aabb'][1])]);relative=combined-eef
 lo=g['oven_center']+OPEN_LO;hi=g['oven_center']+OPEN_HI
 if (combined[1,2]-combined[0,2])>=hi[2]-lo[2]-.02:raise RuntimeError('tray+gripper actual vertical footprint cannot fit mouth')
 z=(lo[2]+hi[2]-relative[0,2]-relative[1,2])/2
 # Whole assembly remains35mm before actual mouth while descending. Normal
 # level tray then enters low enough that Panda palm clears the roof.
 y=lo[1]-relative[1,1]-.035
 desired_center=np.array([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,0]);x=desired_center[0]-offset[0]
 target=np.array([x,desired_center[1]-offset[1],z]);staging_high=np.array([x,y,1.23]);staging_low=np.array([x,y,z])
 details['normal_path_geometry']={'offline_only':True,'mouth_actual_world_lo_m':lo.tolist(),'mouth_actual_world_hi_m':hi.tolist(),'actual_inner_dimensions_m':(hi-lo).tolist(),'held_assembly_relative_eef_aabb_m':relative.tolist(),'initial_combined_vertical_margin_m':float((hi[2]-lo[2]-(combined[1,2]-combined[0,2]))/2),'staging_whole_assembly_front_gap_m':.035,'level_insertion_eef_goal_m':target.tolist(),'staging_high_m':staging_high.tolist(),'staging_low_m':staging_low.tolist(),'real_payload_relative_eef_m':offset.tolist(),'no_geometry_resize_or_wall_change':True}
 return staging_high,staging_low,target
def _verify_payload(scene,details):
 g=_geometry(scene);box=g['tray_oven_aabb'];inside=bool(np.all(box[0]>OPEN_LO)and np.all(box[1]<OPEN_HI));grip=g['gripper_oven_aabb'];zmargin=[float(grip[0,2]-OPEN_LO[2]),float(OPEN_HI[2]-grip[1,2])];details['actual_full_payload_final']={'offline_only':True,'tray_local_aabb_m':box.tolist(),'opening_local_aabb_m':[OPEN_LO.tolist(),OPEN_HI.tolist()],'full_tray_inside_actual_cavity':inside,'gripper_local_aabb_m':grip.tolist(),'gripper_vertical_clearance_m':zmargin,'tray_clearance_each_face_m':np.minimum(box[0]-OPEN_LO,OPEN_HI-box[1]).tolist()}
 if not inside:raise RuntimeError('actual full tray failed physical cavity containment')
class RecordedScene(AluminumTrayScene):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.path_samples=[];self.oven_collision_samples=[];records.append(self)
 def step(self,action):
  result=super().step(action);sim=self.backend.sim;contacts=[]
  for contact in sim.data.contact[:sim.data.ncon]:
   aa=sim.model.geom_id2name(contact.geom1)or'';bb=sim.model.geom_id2name(contact.geom2)or''
   if 'microwave_1'in(aa+bb)and ('steel_thermos_1'in(aa+bb)or'gripper0_'in(aa+bb)or'robot0_'in(aa+bb)) and contact.dist<=.001:
    contacts.append(dict(a=aa,b=bb,dist_m=float(contact.dist),position_m=np.asarray(contact.pos).tolist()))
  if contacts:self.oven_collision_samples.append(dict(time_s=float(sim.data.time),contacts=contacts))
  if self.grasp_assist.held and self.sequence%4==0:
   g=_geometry(self);self.path_samples.append(dict(time_s=float(sim.data.time),tray_local_aabb_m=g['tray_oven_aabb'].tolist(),gripper_local_aabb_m=g['gripper_oven_aabb'].tolist(),eef_world_m=np.asarray(self.raw['robot0_eef_pos']).tolist()))
  return result
class NormalPath(ast.NodeTransformer):
 def visit_For(self,node):
  if ast.unparse(node.iter)=='range(12)':node.iter=ast.parse('range(0)',mode='eval').body
  return self.generic_visit(node)
 def visit_Expr(self,node):
  self.generic_visit(node)
  if not isinstance(node.value,ast.Call)or not isinstance(node.value.func,ast.Name)or node.value.func.id!='move'or len(node.value.args)<2 or not isinstance(node.value.args[1],ast.Constant):return node
  stage=node.value.args[1].value
  if stage=='front_staging':
   node.value.args[0]=ast.Name('staging_high',ast.Load());return[*ast.parse('staging_high,staging_low,insertion_target=_prepare_path(scene,details)').body,node]
  if stage=='lower_to_open_front':node.value.args[0]=ast.Name('staging_low',ast.Load())
  if stage=='insert_into_cavity':
   node.value.args[0]=ast.Name('insertion_target',ast.Load())
   for kw in node.value.keywords:
    if kw.arg=='allow_commit':kw.value=ast.Constant(False)
   return [node,*ast.parse('_verify_payload(scene,details)').body]
  return node
namespace=dict(shared.__dict__);namespace.update(_prepare_path=_prepare_path,_verify_payload=_verify_payload)
tree=NormalPath().visit(ast.parse(Path(shared.__file__).read_text()));exec(compile(ast.fix_missing_locations(tree),str(shared.__file__)+'[normal-tray-mouth-path]','exec'),namespace);namespace.update(R3TabletopFacade=RecordedScene,Evaluator=Evaluator,JUDGE=JUDGE,REASON=REASON)
def run_case(seed,case):
 before=len(records);r=namespace['run_case'](seed,case);scene=records[before];r.update(instance='R3-06',instruction=INSTRUCTION,kind='new real physical CPU engineering script; no model',actual_mouth_path_samples=scene.path_samples,actual_robot_tray_oven_contacts=scene.oven_collision_samples)
 if case=='naive'and scene.oven_collision_samples:r['passed']=False;r['qualification_extra_failure']='new path contacted oven; do not force past mouth'
 del records[before:];return r
