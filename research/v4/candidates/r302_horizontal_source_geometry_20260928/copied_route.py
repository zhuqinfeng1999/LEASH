"""One own CPU constructor/copy path, zero live task controls or models."""
import json,time,hashlib,types,itertools
from pathlib import Path
from types import SimpleNamespace
import numpy as np,mujoco
from scipy.spatial.transform import Rotation,Slerp
from research.v4.candidates.r3_02_oven_only_prefix_20260928.scene import OvenOnlySoupScene
from research.v4.candidates.r3_05_oven_only_prefix_20260928.root_prefix import contact_rows as original_contacts
from research.v4.accelerated.slot_preflight import _ik
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
def main():
 if(H/'copied_route.json').exists():raise ValueError('First copied path only; preserve failure')
 geo=json.loads((H/'source_geometry_complete.json').read_text())
 if not geo['passed']:raise RuntimeError('Full source geometry not clear; no constructor')
 pins=json.loads((H/'source_freeze.json').read_text())['sha256']
 if any(hashlib.sha256((Path(n)if n.startswith('/')else ROOT/n).read_bytes()).hexdigest()!=v for n,v in pins.items()):raise RuntimeError('Source admission drift')
 rows=[];failure=None;started=time.monotonic();s=OvenOnlySoupScene(0);live=s.backend.sim.data._data;model=s.backend.sim.model._model
 state=live.qpos.copy();data=mujoco.MjData(model);data.qpos[:]=live.qpos;mujoco.mj_forward(model,data)
 center=s.grasp_assist._center;body=model.body(s.backend.objects_dict[s.item_name].root_body).id;ov=model.body(s.backend.objects_dict[s.oven_name].root_body).id;pj=model.joint(s.backend.objects_dict[s.item_name].joints[0]).id;adr=int(model.jnt_qposadr[pj]);zone=model.site('steel_thermos_1_v42_grasp').id if 'steel_thermos_1_v42_grasp'in [model.site(i).name for i in range(model.nsite)]else s.grasp_assist._entries[0][2]
 fids=[model.joint('gripper0_right_finger_joint'+str(i)).id for i in(1,2)]
 # Actual original contact helper only literal payload-prefix changes.
 if original_contacts.__code__.co_consts.count('fork_bowl_1_')!=1:raise RuntimeError('Original contact payload prefix drift')
 code=original_contacts.__code__.replace(co_consts=tuple('steel_thermos_1_'if x=='fork_bowl_1_'else x for x in original_contacts.__code__.co_consts))
 contacts=types.FunctionType(code,original_contacts.__globals__)
 proxy=SimpleNamespace(model=s.backend.sim.model,data=SimpleNamespace(qpos=data.qpos))
 home=data.site_xpos[center].copy();homeR=data.site_xmat[center].reshape(3,3).copy();R=np.array(geo['EEF_R']);grip=np.array(geo['EEF_target']);baseBody=data.xpos[body].copy();baseBR=data.xmat[body].reshape(3,3).copy();rel=None
 def pose(phase,t,p,rot,held=False):
  nonlocal failure,rel
  ik=_ik(proxy,p,rot,max_iter=200,position_tol=.012,rotation_tol=.03);row=dict(phase=phase,t=float(t),goal=p.tolist(),ik=ik)
  if not ik['reachable']:rows.append(row);failure=dict(phase=phase,kind='original copied IK not reachable',detail=ik);return False
  if ik.get('restart')!=0:
   rows.append(row);failure=dict(phase=phase,kind='original IK midpoint restart lacks continuous-path proof',detail=ik);return False
  for i,q in enumerate(ik['joint_qpos_rad'],1):data.qpos[model.jnt_qposadr[model.joint('robot0_joint'+str(i)).id]]=q
  mujoco.mj_forward(model,data)
  E=data.site_xmat[center].reshape(3,3).copy();eef=data.site_xpos[center].copy()
  if held:
   data.qpos[adr:adr+3]=eef+E@rel[0];data.qpos[adr+3:adr+7]=Rotation.from_matrix(E@rel[1]).as_quat()[[3,0,1,2]];mujoco.mj_forward(model,data)
  view=SimpleNamespace(backend=SimpleNamespace(sim=SimpleNamespace(model=SimpleNamespace(_model=model),data=SimpleNamespace(_data=data))))
  cr=contacts(view);bad=[x for x in cr if x['non_grasp_hard_stop']]
  hs=model.site('microwave_1_heating_region').id;local=data.site_xmat[hs].reshape(3,3).T@(data.xpos[body]-data.site_xpos[hs]);critical=bool(held and(np.abs(local)<model.site_size[hs]).all())
  row.update(actualEEF=eef.tolist(),contacts=cr,qpos=data.qpos.tolist(),critical=critical)
  rows.append(row)
  if bad:failure=dict(phase=phase,kind='actual copied non-grasp hard overlap>1mm',contacts=bad);return False
  return True
 def segment(phase,start,end,startR,endR,held=False):
  angle=Rotation.from_matrix(endR@startR.T).magnitude();n=max(1,int(np.ceil(np.linalg.norm(end-start)/.025)),int(np.ceil(angle/.1)));interp=Slerp([0,1],Rotation.from_matrix(np.array([startR,endR])))
  for t in np.linspace(0,1,n+1):
   if not pose(phase,t,start*(1-t)+end*t,interp(t).as_matrix(),held):return False
   if rows[-1]['critical']:return True
  return True
 report=dict(constructors=1,Scene_controls=0,GPU=False,models=0,source_unchanged=True,copied_data_only=True)
 try:
  data.qpos[model.jnt_qposadr[fids[0]]]=.04;data.qpos[model.jnt_qposadr[fids[1]]]=-.04;mujoco.mj_forward(model,data)
  above=grip-R[:,2]*.12;above[2]=1.15
  if segment('own_home_to_source_outside_above',home,above,homeR,homeR)and segment('one_local_horizontal_orientation',above,above,homeR,R):
   outside=above.copy();outside[2]=grip[2]
   if segment('horizontal_outside_descent',above,outside,R,R)and segment('ordinary_horizontal_approach',outside,grip,R,R):
    for jid,q in zip(fids,geo['actual_legal_contact_finger_qpos_proposal']):data.qpos[model.jnt_qposadr[jid]]=q
    mujoco.mj_forward(model,data)
    site=s.grasp_assist._entries[0][2];left=data.site_xpos[s.grasp_assist._left];right=data.site_xpos[s.grasp_assist._right];v=right-left;E=data.site_xmat[center].reshape(3,3);eef=data.site_xpos[center].copy();local=E.T@(data.site_xpos[site]-eef);along=float((data.site_xpos[site]-left)@(v/np.linalg.norm(v)))
    report['closure_eligibility']=dict(actual_zone_local=local.tolist(),along=along,gap=float(np.linalg.norm(v)),same_frozen_tol=[.015,.015,.020])
    if not ((abs(local)<=np.array([.015,.015,.020])).all()and 0<=along<=np.linalg.norm(v)):failure=dict(phase='copied_closure',kind='original assist eligibility not met')
    else:
     rel=(E.T@(data.xpos[body]-eef),E.T@data.xmat[body].reshape(3,3));lift=eef.copy();lift[2]=1.23
     if segment('held_ordinary_lift',eef,lift,R,R,True):
      oR=data.xmat[ov].reshape(3,3).copy();op=data.xpos[ov].copy();hs=model.site('microwave_1_heating_region').id;half=model.site_size[hs].copy();hp=data.site_xpos[hs].copy();hR=data.site_xmat[hs].reshape(3,3).copy()
      g3=model.geom('microwave_1_g3').id;g4=model.geom('microwave_1_g4').id;mx=(model.geom_pos[g3,0]+model.geom_size[g3,0]+model.geom_pos[g4,0]-model.geom_size[g4,0])/2
      bp=data.xpos[body].copy();bLocal=oR.T@(bp-op);bLocal[0]=mx;aligned=op+oR@bLocal-R@rel[0]
      targetLocal=hR.T@(bp-hp);targetLocal[0]=mx-(oR.T@(hp-op))[0];targetLocal[1]=-half[1]+.004;target=hp+hR@targetLocal;tl=oR.T@(target-op);tl[2]=.0432+.0575+.005;target=op+oR@tl-R@rel[0]
      descent=aligned.copy();descent[2]=target[2]
      report['fixed_tail']=dict(aligned=aligned.tolist(),outside=descent.tolist(),first_root_critical=target.tolist(),no_release_fullcontainment_claim=True)
      if segment('held_high_mouth_lateral_alignment',lift,aligned,R,R,True)and segment('held_outside_descent',aligned,descent,R,R,True):segment('horizontal_first_original_critical',descent,target,R,R,True)
  if failure is None and not any(r.get('critical')for r in rows):failure=dict(phase='end',kind='original first-critical not reached')
 except Exception as e:failure=dict(phase='instrumentation',kind=type(e).__name__,error=str(e))
 finally:
  report.update(passed=failure is None,failure=failure,rows=rows,live_qpos_unchanged=bool(np.array_equal(state,live.qpos)),wall_s=time.monotonic()-started,virtual_held_not_actual_attach=True)
  (H/'copied_route.json').write_text(json.dumps(report,indent=2)+'\n');s.close()
 print(json.dumps({k:report[k]for k in('passed','failure','wall_s','live_qpos_unchanged')}))
if __name__=='__main__':main()
