"""One copied source-axis horizontal grasp and floor-derived low entry; no controls."""
import json,hashlib,itertools,inspect,textwrap,time,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np,mujoco
from scipy.spatial.transform import Rotation,Slerp
from research.v4.accelerated.slot_preflight import _ik
from research.v4.accelerated.grasp_assist import GraspAssist
from research.v4.candidates.r3_04_horizontal_side_grasp_copy_20260928.scene import HorizontalSideFoilScene,SIDE_POINT
from research.v4.candidates.r3_04_tall60_own_entry_20260928.prefix import selected_contacts
from research.v4.candidates.r3_04_source_axis_grasp_proposal_20260928.contacts import bind_actual_compiled_pads,PAD_NAMES
H=Path(__file__).resolve().parent;ROOT=H.parents[3]

def instrumented_ik():
 src=textwrap.dedent(inspect.getsource(_ik));old='                best = (pos_error + .06 * rot_error, pos_error, rot_error, iteration)'
 if src.count(old)!=1:raise ValueError('Existing IK best-state source seam drift')
 src=src.replace(old,old+'\n                best_qpos = data.qpos.copy()')
 old2='"reason": "ik_no_convergence_or_joint_limit"}'
 if src.count(old2)!=1:raise ValueError('Existing IK failure receipt source seam drift')
 src=src.replace(old2,'"reason": "ik_no_convergence_or_joint_limit", "best_full_qpos": best_qpos.tolist(), "joint_qpos_addresses": qidx.tolist(), "joint_limits": limits.tolist()}')
 ns=dict(_ik.__globals__);exec(compile(src,str(H/'check.py')+'::existing-IK-readonly-failed-qpos-instrument','exec'),ns);return ns['_ik'],src

def main():
 if (H/'report.json').exists():raise ValueError('Only one copied trial permitted')
 for name,sha in json.loads((H/'source_freeze.json').read_text()).items():
  p=Path(name);p=p if p.is_absolute()else ROOT/p
  if hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('Exact source admission drift '+name)
 start=time.monotonic();report={'instance':'R3-04','actual_controls':0,'Scene_CPU_constructors':0,'GPU_models_services':0,'old387_prefix_reused':False,'episode_bounds_unchanged':[1200,90],'native_ready':False,'launch_authorized':False};rows=[];failure=None;scene=None
 try:
  scene=HorizontalSideFoilScene(0);report['Scene_CPU_constructors']=1;m=scene.backend.sim.model._model;d=mujoco.MjData(m);d.qpos[:]=scene.backend.sim.data._data.qpos;d.qvel[:]=0;mujoco.mj_forward(m,d)
  center=scene.grasp_assist._center;body=m.body(scene.backend.objects_dict[scene.item_name].root_body).id;oven=m.body(scene.backend.objects_dict[scene.oven_name].root_body).id;pj=m.joint(scene.backend.objects_dict[scene.item_name].joints[0]).id;pa=int(m.jnt_qposadr[pj]);zone=m.site('v42_foil_sandwich_1_grasp_zone').id
  if not np.array_equal(m.site_pos[zone],np.array(SIDE_POINT)):raise ValueError('Exact side region not compiled')
  records={}
  for name in PAD_NAMES:
   g=m.geom(name).id;records[name]={k:np.asarray(getattr(m,k)[g]).tolist()for k in ['geom_type','geom_size','geom_solref','geom_contype','geom_conaffinity']}
  classify=bind_actual_compiled_pads(records);report['actual_compiled_pad_records']=records
  assist=GraspAssist(m,d,**scene.backend.grasp_assist_kwargs);contact=selected_contacts();ik,src=instrumented_ik();(H/'instrumented_ik_source.py.txt').write_text(src)
  arm=[m.joint(f'robot0_joint{i}').id for i in range(1,8)];fingers=[m.joint('gripper0_right_finger_joint1').id,m.joint('gripper0_right_finger_joint2').id]
  for j,q in zip(fingers,[.04,-.04]):d.qpos[m.jnt_qposadr[j]]=q
  mujoco.mj_forward(m,d);proxy=SimpleNamespace(model=scene.backend.sim.model,data=SimpleNamespace(qpos=d.qpos))
  def cc():
   raw=contact(SimpleNamespace(backend=SimpleNamespace(sim=SimpleNamespace(model=SimpleNamespace(_model=m),data=SimpleNamespace(_data=d)))))
   return [dict(c,exact_pad_classification=classify(c))for c in raw]
  def bad(cs):return[c for c in cs if c['exact_pad_classification']['non_grasp_hard_stop']]
  initial=cc();report['initial_contacts']=initial
  if bad(initial):raise ValueError('Initial >1mm exact nonpad hard overlap')
  sourceR=d.xmat[body].reshape(3,3).copy();R0=np.column_stack(([0.,1.,0.],[0.,0.,1.],[1.,0.,0.]));sourceEEF=sourceR@R0;retreat=.001821250654;grip=d.site_xpos[zone].copy()-sourceR[:,0]*retreat;home=d.site_xpos[center].copy();homeR=d.site_xmat[center].reshape(3,3).copy();pre=grip-sourceR[:,0]*.06
  report.update(actual_source_body=d.xpos[body].tolist(),actual_source_R=sourceR.tolist(),source_EEF_R=sourceEEF.tolist(),original_zone_world=d.site_xpos[zone].tolist(),single_side_grasp_goal=grip.tolist(),single_preapproach_goal=pre.tolist(),home_EEF=home.tolist(),derived_extra_source_backward_m=retreat)
  rel=None;closed_q=.0345
  def pose(phase,t,p,R,*,source=False,q=.04,held=False):
   nonlocal failure
   ans=ik(proxy,p,R,max_iter=200,position_tol=.0005 if source else .012,rotation_tol=math.radians(.5)if source else .03);row={'phase':phase,'t':float(t),'goal':np.asarray(p).tolist(),'wanted_R':R.tolist(),'ik':ans,'private_source_precision':source}
   if not ans['reachable']:
    if 'best_full_qpos'in ans:d.qpos[:]=ans['best_full_qpos'];mujoco.mj_forward(m,d);row.update(failed_best_EEF=d.site_xpos[center].tolist(),failed_best_contacts=cc(),failed_best_qpos=d.qpos.tolist())
    rows.append(row);failure={'kind':'Original solver no convergence at selected source/path bound','phase':phase,'t':float(t),'ik':ans};return False
   for j,v in zip(arm,ans['joint_qpos_rad']):d.qpos[m.jnt_qposadr[j]]=v
   for j,v in zip(fingers,[closed_q if held else q,-closed_q if held else -q]):d.qpos[m.jnt_qposadr[j]]=v
   mujoco.mj_forward(m,d)
   if held:
    A=d.site_xmat[center].reshape(3,3);d.qpos[pa:pa+3]=d.site_xpos[center]+A@rel[0];d.qpos[pa+3:pa+7]=Rotation.from_matrix(A@rel[1]).as_quat()[[3,0,1,2]];mujoco.mj_forward(m,d)
   cs=cc();row.update(actual_EEF=d.site_xpos[center].tolist(),actual_R=d.site_xmat[center].reshape(3,3).tolist(),actual_payload=d.xpos[body].tolist(),actual_qpos=d.qpos.tolist(),actual_contacts=cs,private_zone_eligible=bool(assist._eligible(zone,(.015,.015,.020))),virtual_held=held)
   rows.append(row)
   if bad(cs):failure={'kind':'actual copied hardfinger/palm/environment >1mm','phase':phase,'t':float(t),'contacts':bad(cs)};return False
   return True
  def segment(name,a,b,Ra,Rb,*,source=False,held=False):
   N=max(1,int(np.ceil(np.linalg.norm(b-a)/.025)),int(np.ceil(np.linalg.norm(Rotation.from_matrix(Rb@Ra.T).as_rotvec())/math.radians(5))));interp=Slerp([0,1],Rotation.from_matrix(np.stack([Ra,Rb])))
   for t in np.linspace(0,1,N+1):
    if not pose(name,t,a*(1-t)+b*t,interp(float(t)).as_matrix(),source=source,held=held):return False
   return True
  if segment('home_to_horizontal_side_preapproach',home,pre,homeR,sourceEEF,source=True)and segment('material_axis_source_approach',pre,grip,sourceEEF,sourceEEF,source=True):
   if not rows[-1]['private_zone_eligible']:raise ValueError('Strict source pose outside actual pad/tolerance eligibility')
   for q in np.linspace(.04,closed_q,4):
    if not pose('material_first_closure',float(q),grip,sourceEEF,source=True,q=float(q)):break
   if failure is None:
    A=d.site_xmat[center].reshape(3,3).copy();origin=d.site_xpos[center].copy();rel=(A.T@(d.xpos[body]-origin),A.T@d.xmat[body].reshape(3,3));report['actual_copied_closure_relative']={'position':rel[0].tolist(),'R':rel[1].tolist(),'private_offline_only':True,'not_an_actual_weld_or_success':True}
    def vertices(g):
     Q=d.geom_xmat[g].reshape(3,3);p=d.geom_xpos[g];typ=int(m.geom_type[g])
     if typ==6:return np.array([p+Q@(m.geom_size[g]*v)for v in itertools.product((-1,1),repeat=3)])
     if typ==7:
      mid=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mid]);N=int(m.mesh_vertnum[mid]);return p+(Q@m.mesh_vert[start:start+N].T).T
     raise ValueError('Unknown physical mesh shape; no guessed extent '+str(typ))
    solids=[g for g in range(m.ngeom)if(m.geom(g).name or '').startswith(('foil_sandwich_1_','gripper0_'))and(m.geom_contype[g]or m.geom_conaffinity[g])];V=np.concatenate([vertices(g)for g in solids]);localhand=(V-origin)@A
    op=d.xpos[oven].copy();oR=d.xmat[oven].reshape(3,3).copy();Rentry=np.column_stack((-oR[:,0],oR[:,2],oR[:,1]));entryoffset=(Rentry@localhand.T).T
    floor=m.geom('microwave_1_g7').id;roof=m.geom('microwave_1_g6').id;floorZ=vertices(floor)[:,2].max();roofZ=vertices(roof)[:,2].min();lower=float(floorZ+.005-entryoffset[:,2].min());upper=float(roofZ-.005-entryoffset[:,2].max());carry=max(lower,float(origin[2]+.02))
    report['unique_low_carry']={'floor_world_top':float(floorZ),'roof_world_bottom':float(roofZ),'EEF_floor_roof_5mm_window':[lower,upper],'source_minimum_20mm_lift':float(origin[2]+.02),'selected_lowest_feasible_carry_Z':carry,'solid_geometry':'all actual compiled collision mesh vertices and exact box corners; no hidden item coordinates online','Rentry':Rentry.tolist(),'not_old_1_23_highcarry':True}
    if carry>upper:raise ValueError('Complete hand+payload low floor/roof+lift interval empty')
    lift=origin.copy();lift[2]=carry
    if segment('source_side_low_lift',origin,lift,A,A,held=True)and segment('source_side_clear_single_yaw_alignment',lift,lift,A,Rentry,held=True):
     full=(Rentry@localhand.T).T@oR;left=((vertices(m.geom('microwave_1_g3').id)-op)@oR)[:,0].max();right=((vertices(m.geom('microwave_1_g4').id)-op)@oR)[:,0].min();X=[float(left+.005-full[:,0].min()),float(right-.005-full[:,0].max())];report['full_payload_hand_lateral_EEF_window']=X
     if X[0]>X[1]:raise ValueError('Complete held+hand mouth lateral window empty')
     hs=m.site('microwave_1_heating_region').id;hp=d.site_xpos[hs];Hmat=d.site_xmat[hs].reshape(3,3);bodylocal=oR.T@(hp-op);bodylocal[1]-=m.site_size[hs,1]-.016;bodytarget=op+oR@bodylocal;goal=bodytarget-Rentry@rel[0];localgoal=oR.T@(goal-op);localgoal[0]=sum(X)/2;goal=op+oR@localgoal;goal[2]=carry;outside=lift+oR[:,0]*(sum(X)/2-(oR.T@(lift-op))[0]);outside[2]=carry
     if segment('low_level_mouth_outside_align',lift,outside,Rentry,Rentry,held=True)and segment('horizontal_first_original_critical',outside,goal,Rentry,Rentry,held=True):
      c=Hmat.T@(d.xpos[body]-hp);report['original_critical_copied']=bool(np.all(np.abs(c)<m.site_size[hs]));report['heating_local_body']=c.tolist()
      if not report['original_critical_copied']:raise ValueError('Actual copied original critical not reached')
 except Exception as e:
  if failure is None:failure={'kind':type(e).__name__,'message':str(e)}
 finally:
  report.update(passed=failure is None,failure=failure,rows=rows,elapsed_s=time.monotonic()-start)
  (H/'report.json').write_text(json.dumps(report,indent=2)+'\n')
  if scene is not None:scene.close()
 print(json.dumps({'passed':failure is None,'rows':len(rows),'failure':failure,'elapsed_s':report['elapsed_s']}))
if __name__=='__main__':main()
