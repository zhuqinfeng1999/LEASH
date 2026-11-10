"""Finite zero-control original/plain-cotton actual front/back appearance QA."""
import argparse,json,os,signal,time
from pathlib import Path
import numpy as np,mujoco
from PIL import Image
from research.v4.candidates.panda_glyph_native_pair_20260927 import native_pair as packet
from research.v4.candidates.panda_local_cutout_native_20260927 import native_pair as guard
from . import proof
HERE=proof.HERE
POSES={'shirt_front':{'eye':[-.30,-.65,1.40],'target':[.80,0,.95],'fovy':45.},
       'shirt_back':{'eye':[1.60,.60,1.38],'target':[.76,0,1.00],'fovy':45.}}
def forbid(*a,**k):raise RuntimeError('Zero-task-control appearance fixture forbids step/forward/restore')
def image(s,checks,name,pose):
 with guard.preserved_camera(s,checks,name)as c:
  eye=np.asarray(pose['eye']);target=np.asarray(pose['target']);delta=eye-target
  c.cam.type=mujoco.mjtCamera.mjCAMERA_FREE;c.cam.fixedcamid=-1;c.cam.trackbodyid=-1;c.cam.lookat[:]=target
  c.cam.distance=float(np.linalg.norm(delta));c.cam.azimuth=float(np.degrees(np.arctan2(delta[1],delta[0])));c.cam.elevation=float(-np.degrees(np.arctan2(delta[2],np.linalg.norm(delta[:2]))))
  s.sim.model.vis.global_.fovy=pose['fovy'];frame=np.ascontiguousarray(s.sim.render(width=1920,height=1080,camera_name=None)[::-1])
  actual_eye=np.mean([cam.pos for cam in c.scn.camera],axis=0);actual_forward=np.mean([cam.forward for cam in c.scn.camera],axis=0)
  error=float(np.linalg.norm(actual_eye-eye));forward_error=float(np.linalg.norm(actual_forward+delta/np.linalg.norm(delta)))
  if frame.shape!=(1080,1920,3)or error>1e-5 or forward_error>1e-6:raise RuntimeError('Actual fixed offline appearance pose mismatch')
  checks.append({'name':name+'_actual_pose','eye_error_m':error,'forward_error':forward_error,'actual_eye':actual_eye.tolist()})
  return frame
def run(egl,physical,out):
 if out.exists():raise RuntimeError('Fresh output required; original failures preserved')
 out.mkdir(parents=True);pins=proof.hashes();scenes=[];checks=[];start=time.monotonic();r={'passed':False,'error':None,
  'source_sha256':pins,'simulator_pid':os.getpid(),'requested_physical_gpu':physical,'render_gpu_device_id':egl,
  'controls':0,'model_calls':0,'added_forward':0,'added_restore':0,'rasters_edited':False,
  'appearance_approval':'pending_root_shirt_front_back_and_whole_person_review','denim_label_clearance':'not_assumed',
  'offline_pose_source':'one fixed front and back, no pose search; public cameras unchanged','offline_poses':POSES}
 def expired(*a):raise TimeoutError('Garment native120s')
 prior=signal.signal(signal.SIGALRM,expired);signal.alarm(120)
 try:
  from .check_cpu import semantic
  mapping=json.loads((HERE/'material_partition.json').read_text());models=[];states=[];packets=[];props=[]
  for which in('before','after'):
   facade=proof.initialized(which,True,egl);scenes.append(facade);s=facade.backend;s.step=forbid;s.sim.step=forbid;s.sim.forward=forbid
   base=guard.immutable_snapshot(s)
   with guard.preserved_camera(s,checks,which+'_fresh1'):packet.capture(s)
   p0=packet.props(s)
   with guard.preserved_camera(s,checks,which+'_fresh2'):frames=packet.capture(s)
   p1=packet.props(s)
   if not packet.prop_equal(p0,p1):raise RuntimeError('Actual public proprio not stable')
   with guard.preserved_camera(s,checks,which+'_spectator'):spec=packet.spectator(s)
   packet.save_packet(out/which,frames,spec)
   for name,pose in POSES.items():Image.fromarray(image(s,checks,which+'_'+name,pose)).save(out/which/(name+'_1080.png'))
   guard.same_physics(base,guard.immutable_snapshot(s));models.append(s.sim.model._model);states.append(base);packets.append((frames,spec));props.append(p1)
  r['model_proof']=semantic(models[0],models[1],mapping);r['before_after_physics_exact']=guard.same_physics(states[0],states[1])
  if not packet.prop_equal(props[0],props[1]):raise RuntimeError('Original/plain public proprio differs')
  r['image_differences']=packet.differences(out/'difference_maps',packets[0][0],packets[1][0],packets[0][1],packets[1][1])
  if not all(r['image_differences'][v]['K_exact']and r['image_differences'][v]['T_exact']for v in packet.VIEWS):raise RuntimeError('Public calibrations changed')
  if not all(r['image_differences'][v]['depth']['changed_pixels']==0 for v in packet.VIEWS):raise RuntimeError('Public physical geometry depth differs')
  from research.v4.bridge.run_b0 import observe_physical_gpu
  r['physical_gpu']=observe_physical_gpu(os.getpid(),physical)
  if not r['physical_gpu']['observed']:raise RuntimeError('Physical GPU graphics PID not observed')
  r['passed']=True
 except BaseException as e:r['error']=type(e).__name__+': '+str(e)
 finally:
  for s in scenes:s.close()
  signal.alarm(0);signal.signal(signal.SIGALRM,prior);r['context_checks']=checks;r['wall_seconds']=time.monotonic()-start
  try:r['source_drift']=pins!=proof.hashes()
  except Exception as e:r['source_drift']=True;r['source_drift_error']=str(e)
  (out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 return r
def main():
 p=argparse.ArgumentParser();p.add_argument('--egl-device',type=int,choices=(0,1),required=True);p.add_argument('--physical-gpu',type=int,choices=(0,1),required=True);p.add_argument('--out',type=Path,default=HERE/'native_pair');p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 if a.preflight_only:print(json.dumps({'source_count':len(proof.hashes()),'controls':0,'models':0,'will_construct':False}));return
 os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(a.egl_device))
 r=run(a.egl_device,a.physical_gpu,a.out);print(json.dumps({k:r[k]for k in('passed','error','wall_seconds','source_drift')}));raise SystemExit(0 if r['passed']and not r['source_drift']else 2)
if __name__=='__main__':main()
