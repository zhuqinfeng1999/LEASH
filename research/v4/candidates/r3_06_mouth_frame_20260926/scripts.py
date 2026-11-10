"""Sole correction: exact actual-oven-frame conversion, same normal tray path."""
import ast,json
from pathlib import Path
import numpy as np
from research.v4.candidates.r3_06_mouth_path_20260926 import scripts as original
from research.v4.candidates.r3_03_t1_derived_20260926 import scripts as shared
from research.v4.candidates.r3_06_original_aluminum_tray_20260926.scene import AluminumTrayScene
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTRUCTION=original.INSTRUCTION;OPEN_LO=original.OPEN_LO;OPEN_HI=original.OPEN_HI
_geometry=original._geometry;_verify_payload=original._verify_payload
records=[]
class RecordedScene(original.RecordedScene):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);records.append(self)
def _prepare_path(scene,details):
 sim=scene.backend.sim;g=_geometry(scene);c=g['oven_center'];R=g['oven_rotation'];eef=(np.asarray(scene.raw['robot0_eef_pos'])-c)@R;obj=scene.backend.objects_dict[scene.item_name];center=(np.asarray(sim.data.body_xpos[sim.model.body_name2id(obj.root_body)])-c)@R;offset=center-eef
 combined=np.array([np.minimum(g['tray_oven_aabb'][0],g['gripper_oven_aabb'][0]),np.maximum(g['tray_oven_aabb'][1],g['gripper_oven_aabb'][1])]);relative=combined-eef
 if combined[1,2]-combined[0,2]>=OPEN_HI[2]-OPEN_LO[2]-.020:raise RuntimeError('tray+gripper true vertical footprint cannot fit mouth')
 z=(OPEN_LO[2]+OPEN_HI[2]-relative[0,2]-relative[1,2])/2
 y=OPEN_LO[1]-relative[1,1]-.035
 desired_center=(OPEN_LO+OPEN_HI)/2;x=desired_center[0]-offset[0]
 local_target=np.array([x,desired_center[1]-offset[1],z]);local_low=np.array([x,y,z]);local_high=np.array([x,y,(1.23-c[2]-R[2,0]*x-R[2,1]*y)/R[2,2]])
 predicted=relative+local_target
 xz_margin=np.minimum(predicted[0]-OPEN_LO,OPEN_HI-predicted[1])[[0,2]]
 if np.any(xz_margin<.010):raise RuntimeError('complete tray+gripper predicted X/Z footprint does not fit mouth with10mm margin')
 target=c+R@local_target;high=c+R@local_high;low=c+R@local_low
 details['normal_path_geometry']={'offline_only':True,'mouth_actual_local_lo_m':OPEN_LO.tolist(),'mouth_actual_local_hi_m':OPEN_HI.tolist(),'actual_inner_dimensions_m':(OPEN_HI-OPEN_LO).tolist(),'actual_oven_origin_world_m':c.tolist(),'actual_oven_rotation':R.tolist(),'held_assembly_relative_eef_local_aabb_m':relative.tolist(),'predicted_complete_assembly_at_insertion_local_aabb_m':predicted.tolist(),'predicted_complete_assembly_xz_clearance_m':xz_margin.tolist(),'staging_whole_assembly_front_gap_m':.035,'level_insertion_eef_goal_world_m':target.tolist(),'staging_high_world_m':high.tolist(),'staging_low_world_m':low.tolist(),'real_payload_relative_eef_local_m':offset.tolist(),'no_geometry_resize_or_wall_change':True}
 return high,low,target
namespace=dict(shared.__dict__);namespace.update(_prepare_path=_prepare_path,_verify_payload=_verify_payload)
tree=original.NormalPath().visit(ast.parse(Path(shared.__file__).read_text()));exec(compile(ast.fix_missing_locations(tree),str(shared.__file__)+'[exact-oven-frame-normal-tray-mouth-path]','exec'),namespace);namespace.update(R3TabletopFacade=RecordedScene,Evaluator=original.Evaluator,JUDGE=original.JUDGE,REASON=original.REASON)
def run_case(seed,case):
 before=len(records);parent_before=len(original.records);r=namespace['run_case'](seed,case);scene=records[before];r.update(instance='R3-06',instruction=INSTRUCTION,kind='new real physical CPU engineering script; no model',actual_mouth_path_samples=scene.path_samples,actual_robot_tray_oven_contacts=scene.oven_collision_samples)
 if case=='naive'and scene.oven_collision_samples:r['passed']=False;r['qualification_extra_failure']='new path contacted oven; do not force past mouth'
 del records[before:];del original.records[parent_before:];return r
