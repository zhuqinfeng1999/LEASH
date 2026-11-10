"""Zero-control before/after native proof for the local letter-hole fill."""
import argparse,json,os,signal,time
from contextlib import contextmanager
from pathlib import Path
import numpy as np,mujoco
from PIL import Image
from . import proof
from research.v4.candidates.panda_glyph_native_pair_20260927 import native_pair as packet
from research.v4.candidates.panda_glyph_close_native_20260927.derive import world_mesh
HERE=proof.HERE;ROOT=proof.ROOT
FIT=json.loads((ROOT/'research/v4/candidates/panda_glyph_close_native_20260927/camera_fit.json').read_text());POSE=FIT['pose']
CAM_FIELDS=('type','fixedcamid','trackbodyid','lookat','distance','azimuth','elevation')

def context_values(s):
 c=s.sim._render_context_offscreen
 if c is None:raise RuntimeError('Existing normal constructor context required')
 return {k:np.asarray(getattr(c.cam,k)).copy() for k in CAM_FIELDS}|{'global_fovy':np.asarray(s.sim.model.vis.global_.fovy).copy(),'geomgroup':np.asarray(c.vopt.geomgroup).copy()}

def context_equal(a,b):return set(a)==set(b)and all(packet.exact(a[k],b[k])for k in a)

@contextmanager
def preserved_camera(s,checks,name):
 old=context_values(s);base=packet.state(s);c=s.sim._render_context_offscreen
 try:yield c
 finally:
  for k in CAM_FIELDS:
   if k=='lookat':c.cam.lookat[:]=old[k]
   else:setattr(c.cam,k,old[k].item())
  s.sim.model.vis.global_.fovy=float(old['global_fovy']);c.vopt.geomgroup[:]=old['geomgroup']
  row={'name':name,'context_restored_exact':context_equal(old,context_values(s)),'integration_exact':packet.exact(base,packet.state(s))}
  checks.append(row)
  if not all(row[k]for k in('context_restored_exact','integration_exact')):raise RuntimeError('Camera/state changed '+name)

def close(s,checks):
 m,d=s.sim.model._model,s.sim.data._data;v=world_mesh(m,d,45)
 if m.geom(45).name!=FIT['glyph_geom_name']or not np.array_equal(d.qpos,np.asarray(FIT['actual_qpos']))or not np.allclose(v.mean(axis=0),FIT['actual_world_centroid_m'],rtol=0,atol=1e-10):raise RuntimeError('Frozen actual glyph/state no longer matches camera derivation')
 with preserved_camera(s,checks,'close_1080')as c:
  c.cam.type=mujoco.mjtCamera.mjCAMERA_FREE;c.cam.fixedcamid=-1;c.cam.trackbodyid=-1;c.cam.lookat[:]=POSE['look_at_world_m'];c.cam.distance=POSE['distance_m'];c.cam.azimuth=POSE['azimuth_deg'];c.cam.elevation=POSE['elevation_deg'];s.sim.model.vis.global_.fovy=POSE['fovy_deg']
  img=np.ascontiguousarray(s.sim.render(width=1920,height=1080,camera_name=None)[::-1]);eye=np.mean([x.pos for x in c.scn.camera],axis=0);forward=np.mean([x.forward for x in c.scn.camera],axis=0);target=np.asarray(POSE['position_world_m']);delta=target-np.asarray(POSE['look_at_world_m']);e=float(np.linalg.norm(eye-target));f=float(np.linalg.norm(forward+delta/np.linalg.norm(delta)))
  if img.shape!=(1080,1920,3)or img.dtype!=np.uint8 or e>1e-5 or f>1e-6:raise RuntimeError('Actual close GL pose/resolution mismatch')
  checks.append({'name':'actual_close_pose','eye_error_m':e,'forward_error':f,'actual_eye_world_m':eye.tolist(),'actual_forward':forward.tolist()})
 return img

def snapshot(s,checks):
 with preserved_camera(s,checks,'public640_RGBD_K_T'):frames=packet.capture(s)
 with preserved_camera(s,checks,'normal_spectator1080'):spectator=packet.spectator(s)
 return frames,spectator,close(s,checks)

def forbid(*a,**k):raise RuntimeError('Native fixture forbids extra step/forward/state restore')

def immutable_snapshot(s):
 m,d=s.sim.model._model,s.sim.data._data
 forces={k:np.asarray(getattr(d,k)).copy()for k in('qfrc_bias','qfrc_passive','qfrc_actuator','qfrc_applied','qfrc_constraint','xfrc_applied')};forces['active_efc_force']=np.asarray(d.efc_force[:int(d.nefc)]).copy()
 return {'integration':packet.state(s),'ctrl':np.asarray(d.ctrl).copy(),'qpos':np.asarray(d.qpos).copy(),'qvel':np.asarray(d.qvel).copy(),'act':np.asarray(d.act).copy(),'time':np.asarray(d.time),'camera':{k:np.asarray(getattr(m,k)).copy()for k in('cam_pos','cam_quat','cam_fovy','cam_bodyid')},'contacts':proof.active_contacts(s),'forces':forces}

def same_physics(a,b):
 fields={k:packet.exact(a[k],b[k])for k in('integration','ctrl','qpos','qvel','act','time')};fields['camera']=all(packet.exact(a['camera'][k],b['camera'][k])for k in a['camera']);fields['active_contacts']=a['contacts']==b['contacts'];fields['active_forces']=all(packet.exact(a['forces'][k],b['forces'][k])for k in a['forces'])
 if not all(fields.values()):raise RuntimeError('Same physical state guard failed '+str(fields))
 return fields

def run(egl,physical,out):
 if out.exists():raise RuntimeError('Fresh output only; previous failures cannot be overwritten')
 out.mkdir(parents=True);pins=proof.hashes();start=time.monotonic();scenes=[];checks=[]
 r={'passed':False,'error':None,'source_sha256':pins,'controls':0,'added_sim_forward':0,'added_state_restore':0,'model_calls':0,'simulator_pid':os.getpid(),'render_gpu_device_id':egl,'requested_physical_gpu':physical,'online_protocol_adopted':False,'appearance_approval':'pending_root_actual_before_after_review','intentional_visual_change':'Original dark mesh44 actual letter holes filled; noncollision glyph45 group1 to hidden5. No alpha-only shortcut.'}
 def expired(*a):raise TimeoutError('Native local-fill120s bound')
 prior=signal.signal(signal.SIGALRM,expired);signal.alarm(120)
 try:
  packets={};props={};snapshots={}
  for which in('before','after'):
   s=proof.initialized(which,True,egl);scenes.append(s);s.step=forbid;s.sim.step=forbid;s.sim.forward=forbid
   if not np.array_equal(np.asarray(s.sim._render_context_offscreen.vopt.geomgroup),[0,1,1,0,0,0]):raise RuntimeError('Unexpected existing visual-group rendering option')
   base=immutable_snapshot(s)
   with preserved_camera(s,checks,which+'_fresh1'):packet.capture(s)
   p1=packet.props(s)
   with preserved_camera(s,checks,which+'_fresh2'):packet.capture(s)
   p2=packet.props(s)
   if not packet.prop_equal(p1,p2):raise RuntimeError('Fresh proprioception unstable '+which)
   r.setdefault('fresh_proprio',{})[which]=packet.serialize(p2)
   frames,spec,detail=snapshot(s,checks)
   if not packet.prop_equal(p2,packet.props(s)):raise RuntimeError('Offline render changed proprioception '+which)
   same_physics(base,immutable_snapshot(s));snapshots[which]=base;props[which]=p2;packets[which]=(frames,spec,detail)
   packet.save_packet(out/which,frames,spec);Image.fromarray(detail).save(out/which/'local_panel_close_1080.png')
   np.savez_compressed(out/which/'actual_physics_state.npz',**{k:base[k]for k in('integration','ctrl','qpos','qvel','act','time')},**base['forces']);(out/which/'active_contacts.json').write_text(json.dumps(base['contacts'],indent=2)+'\n')
   r.setdefault('constructors',{})[which]=s._brand_compile_calls
  r['model_proof']=proof.semantic(scenes[0].sim.model._model,scenes[1].sim.model._model)
  r['before_after_physics_exact']=same_physics(snapshots['before'],snapshots['after'])
  if not packet.prop_equal(props['before'],props['after']):raise RuntimeError('Fresh before/after public proprio differs')
  r['before_after_proprio_exact']=True
  from research.v4.bridge.run_b0 import observe_physical_gpu
  r['physical_gpu']=observe_physical_gpu(os.getpid(),physical)
  if not r['physical_gpu']['observed']:raise RuntimeError('Actual physical graphics PID not observed')
  b,a=packets['before'],packets['after'];r['image_differences']=packet.differences(out/'difference_maps',b[0],a[0],b[1],a[1]);delta,mask,mag=packet.image_delta(b[2],a[2]);r['image_differences']['local_panel_close']=delta;Image.fromarray(mask.astype(np.uint8)*255).save(out/'difference_maps/local_close_changed_pixels.png');Image.fromarray(mag).save(out/'difference_maps/local_close_max_delta.png')
  if not all(r['image_differences'][v]['K_exact']and r['image_differences'][v]['T_exact']for v in packet.VIEWS):raise RuntimeError('Public calibration differs')
  r.update(passed=True,context_checks=checks,close_pose=POSE,notice_preserved=True)
 except BaseException as e:r['error']=type(e).__name__+': '+str(e)
 finally:
  for s in scenes:s.close()
  signal.alarm(0);signal.signal(signal.SIGALRM,prior);r.update(context_checks=checks,wall_seconds=time.monotonic()-start)
  try:r['source_drift']=pins!=proof.hashes()
  except Exception as e:r['source_drift']=True;r['source_drift_error']=str(e)
  (out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 return r

def main():
 p=argparse.ArgumentParser();p.add_argument('--egl-device',type=int,choices=(0,1),required=True);p.add_argument('--physical-gpu',type=int,choices=(0,1),required=True);p.add_argument('--out',type=Path,default=HERE/'native_pair');p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 if a.preflight_only:print(json.dumps({'source_count':len(proof.hashes()),'controls':0,'will_construct':False,'models':0}));return
 os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(a.egl_device));r=run(a.egl_device,a.physical_gpu,a.out);print(json.dumps({k:r[k]for k in('passed','error','wall_seconds','source_drift')}));raise SystemExit(0 if r['passed']and not r['source_drift']else 2)
if __name__=='__main__':main()
