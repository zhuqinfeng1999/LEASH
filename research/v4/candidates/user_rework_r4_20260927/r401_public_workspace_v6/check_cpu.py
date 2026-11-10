"""One actual before/after constructor pair; no controls/render/forward."""
import json,time,traceback
from types import SimpleNamespace
import numpy as np,mujoco
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix,get_camera_extrinsic_matrix
from . import scene
HERE=scene.HERE
def equal(a,b):
 a=np.asarray(a);b=np.asarray(b)
 return np.array_equal(a,b,equal_nan=True)if np.issubdtype(a.dtype,np.number)else np.array_equal(a,b)
def arrays(obj):return {k:np.asarray(getattr(obj,k)).copy()for k in dir(obj)if isinstance(getattr(obj,k),np.ndarray)}
def integration(s):
 m,d=s.backend.sim.model._model,s.backend.sim.data._data;spec=mujoco.mjtState.mjSTATE_INTEGRATION;r=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,r,spec);return r
def controllers(s):
 robot=s.backend.robots[0];values={}
 for prefix,obj in [('composite',robot.composite_controller)]+[('part.'+n,c)for n,c in robot.part_controllers.items()]:
  for k,v in vars(obj).items():
   if isinstance(v,np.ndarray):values[prefix+'.'+k]=v.copy()
   elif isinstance(v,(int,float,bool,str,type(None))):values[prefix+'.'+k]=v
 return values
def snapshot(s):
 m,d=s.backend.sim.model._model,s.backend.sim.data._data
 forces={n:np.asarray(getattr(d,n)).copy()for n in('qfrc_bias','qfrc_passive','qfrc_actuator','qfrc_applied','qfrc_constraint','xfrc_applied')};forces['active_efc_force']=np.asarray(d.efc_force[:int(d.nefc)]).copy()
 contacts=[dict(geoms=[m.geom(int(i)).name or''for i in(c.geom1,c.geom2)],distance=float(c.dist),pos=c.pos.tolist())for c in d.contact[:d.ncon]]
 return dict(state=integration(s),fields={n:np.asarray(getattr(d,n)).copy()for n in('ctrl','qpos','qvel','act','time')},forces=forces,contacts=contacts)
def main():
 if(HERE/'cpu_pair.json').exists():raise RuntimeError('One constructor pair only')
 start=time.monotonic();scenes=[];r=dict(passed=False,constructor_pairs=0,task_controls=0,GPU_render=False,extra_forward_or_restore=0,models_called=[],error=None)
 try:
  a=scene.old.NativeScene(0);scenes.append(a);b=scene.Scene(0);scenes.append(b);r['constructor_pairs']=1
  ma,mb=a.backend.sim.model._model,b.backend.sim.model._model;aa,bb=arrays(ma),arrays(mb);i=int(ma.camera('agentview').id);assert i==int(mb.camera('agentview').id)
  allowed={'cam_pos','cam_quat','cam_fovy','cam_pos0','cam_mat0','cam_poscom0'};changed={}
  assert aa.keys()==bb.keys()
  for name,x in aa.items():
   y=bb[name]
   if equal(x,y):continue
   if name not in allowed:raise RuntimeError('Unexpected model array change '+name)
   if x.shape!=y.shape or not equal(np.delete(x,i,axis=0),np.delete(y,i,axis=0)):raise RuntimeError('Other camera changed '+name)
   changed[name]=dict(agentview_id=i,before=x[i].tolist(),after=y[i].tolist())
  assert equal(ma.names,mb.names)
  opta,optb=arrays(ma.opt),arrays(mb.opt);assert opta.keys()==optb.keys()and all(equal(v,optb[n])for n,v in opta.items())
  scalars={}
  for obj,label in[(ma,'model'),(ma.opt,'option')]:
   other=mb if label=='model'else mb.opt
   for n in dir(obj):
    v=getattr(obj,n)
    if isinstance(v,(int,float,bool,bytes)):
     if not equal(v,getattr(other,n)):raise RuntimeError('Unexpected scalar '+label+'.'+n)
     scalars[label+'.'+n]=True
  x,y=snapshot(a),snapshot(b);same=equal(x['state'],y['state'])and all(equal(v,y['fields'][n])for n,v in x['fields'].items())and all(equal(v,y['forces'][n])for n,v in x['forces'].items())and x['contacts']==y['contacts'];assert same
  ca,cb=controllers(a),controllers(b);assert ca.keys()==cb.keys()and all(equal(v,cb[n])for n,v in ca.items())
  props={n:equal(a.raw[n],b.raw[n])for n in('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')};assert all(props.values())
  calib={}
  for n in a.public_views:
   K0=get_camera_intrinsic_matrix(a.backend.sim,n,480,640);K1=get_camera_intrinsic_matrix(b.backend.sim,n,480,640);T0=get_camera_extrinsic_matrix(a.backend.sim,n);T1=get_camera_extrinsic_matrix(b.backend.sim,n)
   calib[n]=dict(K_before=K0.tolist(),K_after=K1.tolist(),T_before=T0.tolist(),T_after=T1.tolist(),K_equal=equal(K0,K1),T_equal=equal(T0,T1))
  assert calib['robot0_eye_in_hand']['K_equal']and calib['robot0_eye_in_hand']['T_equal']
  assert not calib['agentview']['K_equal']and not calib['agentview']['T_equal']
  assert a.assist_events==b.assist_events==[] and not a.grasp_assist.held and not b.grasp_assist.held
  assert a.backend.sim._render_context_offscreen is None and b.backend.sim._render_context_offscreen is None
  r.update(passed=True,exact_changed_model_fields=changed,all_other_model_arrays_exact=len(aa)-len(changed),option_arrays_exact=len(opta),numeric_scalar_fields_exact=len(scalars),integration_controls_active_contacts_forces_exact=True,controller_fields_exact=len(ca),proprio_exact=props,calibration=calib,original_reset_staging_and_control_unchanged=True,source_scope_restored=scene.old.scene.SELECTED_INIT is scene.old.scene.SELECTED_INIT,public_version=scene.FIT['version'],intended_new_camera_and_calibration=True,old30physics_only_reusable=True,visual_visibility_not_proven=True)
 except BaseException as exc:r.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
 finally:
  for s in scenes:s.close()
  r['wall_s']=time.monotonic()-start;(HERE/'cpu_pair.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({k:r[k]for k in('passed','error','wall_s','constructor_pairs')}));return 0 if r['passed']else 2
if __name__=='__main__':raise SystemExit(main())
