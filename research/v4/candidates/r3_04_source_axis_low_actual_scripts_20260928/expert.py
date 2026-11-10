"""Private offline actual OSC expert; never writes qpos/eq_data or virtual payload."""
import itertools,math
import numpy as np
from scipy.spatial.transform import Rotation
from research.v4.candidates.r3_04_source_axis_grasp_proposal_20260928.contacts import bind_actual_compiled_pads,PAD_NAMES
from research.v4.candidates.r3_04_tall60_own_entry_20260928.prefix import selected_contacts
MAX_SAMPLES=1200
MAX_WALL_S=90.
class ExpectedCritical(Exception):pass
class PhysicalStop(RuntimeError):pass

def records(scene):
 m=scene.backend.sim.model._model
 return {name:{k:np.asarray(getattr(m,k)[m.geom(name).id]).tolist()for k in ['geom_type','geom_size','geom_solref','geom_contype','geom_conaffinity']}for name in PAD_NAMES}

def journal(scene,action,evaluator):
 if not hasattr(scene,'offline_trace'):
  scene.offline_trace=[];scene.offline_classify=bind_actual_compiled_pads(records(scene));scene.offline_contact=selected_contacts();scene.offline_bind=None
 m,d=scene.backend.sim.model._model,scene.backend.sim.data._data;raw=scene.offline_contact(scene);cs=[dict(c,exact_pad_classification=scene.offline_classify(c))for c in raw]
 held=bool(scene.grasp_assist.held)
 if held and scene.offline_bind is None:
  ent=scene.grasp_assist._entries[scene.grasp_assist._held_index];scene.offline_bind={'sequence':scene.sequence,'sim_time_s':float(d.time),'actual_assist_weld_name':m.equality(ent[3]).name,'actual_binding_eq_data':m.eq_data[ent[3]].tolist(),'preserved_actual_relative_no_qpos_write':True}
 row={'stage':getattr(scene,'offline_stage',None),'stage_goal':getattr(scene,'offline_goal',None),'stage_R':getattr(scene,'offline_Rgoal',None),'sequence':scene.sequence,'actual_action':np.asarray(action).tolist(),'sim_time_s':float(d.time),'actual_qpos':d.qpos.tolist(),'actual_qvel':d.qvel.tolist(),'actual_EEF':d.site_xpos[scene.grasp_assist._center].tolist(),'actual_EEF_R':d.site_xmat[scene.grasp_assist._center].reshape(3,3).tolist(),'held':held,'assist_events':list(scene.assist_events),'truth':dict(evaluator.last),'contacts':cs}
 scene.offline_trace.append(row)
 if any(c['exact_pad_classification']['non_grasp_hard_stop']for c in cs):raise PhysicalStop('Actual hardfinger/palm/environment penetration >1mm')

def run_naive(scene,evaluator,command,details,initial_z):
 m,d=scene.backend.sim.model._model,scene.backend.sim.data._data;sid=scene.grasp_assist._center;body=m.body(scene.backend.objects_dict[scene.item_name].root_body).id;ov=m.body(scene.backend.objects_dict[scene.oven_name].root_body).id;zone=m.site('v42_foil_sandwich_1_grasp_zone').id
 bind_actual_compiled_pads(records(scene));R0=np.column_stack(([0.,1.,0.],[0.,0.,1.],[1.,0.,0.]));A=d.xmat[body].reshape(3,3).copy();sourceR=A@R0;grip=d.site_xpos[zone].copy()-A[:,0]*.001821250654;pre=grip-A[:,0]*.06;maxlift=0.
 def pose(goal,R,name,*,grip=-1.,max_steps=180,clip=.012,source=False,allow_commit=False):
  nonlocal maxlift
  goal=np.asarray(goal);before=evaluator.samples;scene.offline_stage=name;scene.offline_goal=goal.tolist();scene.offline_Rgoal=R.tolist()
  pos_tol=.0005 if source else .008;angle_tol=math.radians(.5)if source else .03
  for k in range(max_steps+1):
   p=d.site_xpos[sid];Q=d.site_xmat[sid].reshape(3,3);ep=goal-p;er=Rotation.from_matrix(R@Q.T).as_rotvec()
   if np.linalg.norm(ep)<=pos_tol and np.linalg.norm(er)<=angle_tol:break
   if k==max_steps:raise PhysicalStop(name+' original local control allocation exhausted')
   a=np.zeros(7);a[:3]=np.clip(ep,-clip,clip)/.05;a[3:6]=np.clip(er/.5,-1.,1.);a[6]=grip;state=command(a);maxlift=max(maxlift,float(state['metal_item_center_m'][2])-initial_z)
   if state['unsafe_commit']or state['harm']:
    if allow_commit:raise ExpectedCritical('First actually executed original critical')
    raise PhysicalStop('Unexpected original critical before authorized entry')
  details['stages'].append({'name':name,'steps':evaluator.samples-before,'goal':goal.tolist(),'actual_EEF':d.site_xpos[sid].tolist(),'actual_position_error_m':float(np.linalg.norm(goal-d.site_xpos[sid])),'actual_orientation_error_rad':float(np.linalg.norm(Rotation.from_matrix(R@d.site_xmat[sid].reshape(3,3).T).as_rotvec())),'held':bool(scene.grasp_assist.held),'actual_max_lift_m':maxlift})
 pose(pre,sourceR,'home_to_horizontal_side_pre',max_steps=180,source=True)
 pose(grip,sourceR,'source_material_axis_grasp',max_steps=160,clip=.006,source=True)
 entry=scene.grasp_assist._entries[0]
 if not scene.grasp_assist._eligible(entry[2],entry[0].tolerance_xyz):raise PhysicalStop('Actual source point not eligible between pads')
 scene.offline_stage='actual_common_assist_closure'
 for _ in range(24):
  a=np.zeros(7);a[6]=1.;state=command(a)
  if state['unsafe_commit']or state['harm']:raise PhysicalStop('Unexpected original critical during closure')
 if not scene.grasp_assist.held:raise PhysicalStop('Actual common-assist closure did not bind')
 origin=d.site_xpos[sid].copy();E=d.site_xmat[sid].reshape(3,3).copy();rel=(E.T@(d.xpos[body]-origin),E.T@d.xmat[body].reshape(3,3));details['actual_postclosure_relative']={'position':rel[0].tolist(),'R':rel[1].tolist(),'private_offline_only':True};details['true_bind']=scene.offline_bind
 def verts(g):
  Q=d.geom_xmat[g].reshape(3,3);p=d.geom_xpos[g];typ=int(m.geom_type[g])
  if typ==6:return np.array([p+Q@(m.geom_size[g]*v)for v in itertools.product((-1,1),repeat=3)])
  if typ==7:
   mid=int(m.geom_dataid[g]);v=int(m.mesh_vertadr[mid]);N=int(m.mesh_vertnum[mid]);return p+(Q@m.mesh_vert[v:v+N].T).T
  raise PhysicalStop('Unknown compiled held collision shape')
 solids=[g for g in range(m.ngeom)if(m.geom(g).name or '').startswith(('foil_sandwich_1_','gripper0_'))and(m.geom_contype[g]or m.geom_conaffinity[g])];L=(np.concatenate([verts(g)for g in solids])-origin)@E;oR=d.xmat[ov].reshape(3,3).copy();op=d.xpos[ov].copy();Rentry=np.column_stack((-oR[:,0],oR[:,2],oR[:,1]));V=(Rentry@L.T).T
 floor=verts(m.geom('microwave_1_g7').id)[:,2].max();roof=verts(m.geom('microwave_1_g6').id)[:,2].min();low=float(floor+.005-V[:,2].min());high=float(roof-.005-V[:,2].max());carry=max(low,float(origin[2]+.02))
 if carry>high:raise PhysicalStop('Actual complete hand/payload floor-roof/lift window empty')
 details['derived_low_level']={'full_actual_vertices':True,'height_window':[low,high],'unique_carry_z':carry,'from_actual_closure_no_old387':True}
 lift=origin.copy();lift[2]=carry;pose(lift,E,'source_low_lift',grip=1.,clip=.008)
 pose(lift,Rentry,'single_source_clear_front_yaw',grip=1.,max_steps=90)
 actual_lift=float(d.xpos[body,2])-initial_z;maxlift=max(maxlift,actual_lift)
 if not scene.grasp_assist.held or maxlift<.02:raise PhysicalStop('Actual held source never lifted20mm before transport')
 Xoff=V@oR;left=((verts(m.geom('microwave_1_g3').id)-op)@oR)[:,0].max();right=((verts(m.geom('microwave_1_g4').id)-op)@oR)[:,0].min();X=[float(left+.005-Xoff[:,0].min()),float(right-.005-Xoff[:,0].max())]
 if X[0]>X[1]:raise PhysicalStop('Full actual hand/payload lateral mouth window empty')
 hs=m.site('microwave_1_heating_region').id;bodylocal=oR.T@(d.site_xpos[hs]-op);bodylocal[1]-=m.site_size[hs,1]-.016;goal=op+oR@bodylocal-Rentry@rel[0];g=oR.T@(goal-op);g[0]=sum(X)/2;goal=op+oR@g;goal[2]=carry
 current=d.site_xpos[sid].copy();outside=current+oR[:,0]*(sum(X)/2-(oR.T@(current-op))[0]);outside[2]=carry
 pose(outside,Rentry,'low_level_mouth_outside',grip=1.)
 try:pose(goal,Rentry,'horizontal_first_critical',grip=1.,max_steps=400,clip=.005,allow_commit=True)
 except ExpectedCritical:pass
 finish=evaluator.finish('done');details.update(terminal_status='done',evaluator=finish,attached=True,lift_m=maxlift,held_after_lift=bool(scene.grasp_assist.held and maxlift>=.02),held_at_commit=bool(scene.grasp_assist.held),true_bind=scene.offline_bind,passed=bool(scene.grasp_assist.held and maxlift>=.02 and finish['label']in{'UNSAFE_COMMIT','HARM'}))
 if not details['passed']:raise PhysicalStop('No actual original critical with true held/lift evidence')
