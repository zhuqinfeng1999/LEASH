"""Finite same-state native appearance proof; no physical controls or source edits."""
import argparse,hashlib,json,os,signal,time
from pathlib import Path
import numpy as np,mujoco
from PIL import Image
from research.v4.candidates.panda_glyph_hide_visual_20260927 import helper
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
FIELDS=('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos');VIEWS=('agentview','robot0_eye_in_hand')
def arrays(o):return{k:getattr(o,k).copy()for k in dir(o)if isinstance(getattr(o,k),np.ndarray)}
def exact(a,b):return np.array_equal(a,b,equal_nan=True)if np.issubdtype(a.dtype,np.number)else np.array_equal(a,b)
def state(s):
 m,d=s.sim.model._model,s.sim.data._data;sig=mujoco.mjtState.mjSTATE_INTEGRATION;x=np.empty(mujoco.mj_stateSize(m,sig));mujoco.mj_getState(m,d,x,sig);return x

def props(s):return{k:np.asarray(s.raw[k]).copy()for k in FIELDS}
def serialize(p):return{k:dict(dtype=str(v.dtype),shape=list(v.shape),value=v.tolist())for k,v in p.items()}
def prop_equal(a,b):return all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and exact(a[k],b[k])for k in FIELDS)
def capture(s):
 from robosuite.utils.camera_utils import get_real_depth_map,get_camera_extrinsic_matrix,get_camera_intrinsic_matrix
 s.raw=s._get_observations(force_update=True);frames={}
 for v in VIEWS:
  rgb=np.ascontiguousarray(np.asarray(s.raw[v+'_image'])[::-1]);depth=np.ascontiguousarray(get_real_depth_map(s.sim,np.asarray(s.raw[v+'_depth']).squeeze())[::-1])
  if rgb.shape!=(480,640,3)or depth.shape!=(480,640):raise RuntimeError('Actual native public640RGBD required')
  frames[v]=dict(rgb=rgb,depth_m=depth,intrinsics=get_camera_intrinsic_matrix(s.sim,v,480,640),extrinsics=get_camera_extrinsic_matrix(s.sim,v))
 return frames

def spectator(s):
 frame=np.ascontiguousarray(s.sim.render(width=1920,height=1080,camera_name='research_spectator')[::-1])
 if frame.shape!=(1080,1920,3)or frame.dtype!=np.uint8:raise RuntimeError('Actual native1080 spectator required, no upscale/fallback')
 return frame

def save_packet(where,frames,spec):
 where.mkdir()
 for v,c in frames.items():
  Image.fromarray(c['rgb']).save(where/(v+'.png'));np.savez_compressed(where/(v+'_RGBD_K_T.npz'),**c)
 Image.fromarray(spec).save(where/'research_spectator_1080.png')

def image_delta(a,b):
 x=a.astype(np.int16)-b.astype(np.int16);mask=np.any(x!=0,axis=-1);return dict(changed_pixels=int(mask.sum()),changed_channels=int(np.count_nonzero(x)),max_abs_delta=int(np.max(np.abs(x)))),mask,np.max(np.abs(x),axis=-1).astype(np.uint8)
def differences(where,before,after,sp0,sp1):
 where.mkdir();summary={}
 for v in VIEWS:
  a,b=before[v],after[v];r,mask,mag=image_delta(a['rgb'],b['rgb']);depth=np.abs(a['depth_m']-b['depth_m']);dm=depth!=0
  Image.fromarray(mask.astype(np.uint8)*255).save(where/(v+'_rgb_changed_pixels.png'));Image.fromarray(mag).save(where/(v+'_rgb_max_delta.png'));Image.fromarray(dm.astype(np.uint8)*255).save(where/(v+'_depth_changed_pixels.png'));Image.fromarray(np.clip(depth/.005*255,0,255).astype(np.uint8)).save(where/(v+'_depth_delta_5mm_white.png'));np.save(where/(v+'_depth_delta_m.npy'),depth)
  summary[v]=dict(rgb=r,depth=dict(changed_pixels=int(dm.sum()),max_abs_delta_m=float(depth.max())),K_exact=exact(a['intrinsics'],b['intrinsics']),T_exact=exact(a['extrinsics'],b['extrinsics']))
 r,mask,mag=image_delta(sp0,sp1);Image.fromarray(mask.astype(np.uint8)*255).save(where/'spectator_changed_pixels.png');Image.fromarray(mag).save(where/'spectator_max_delta.png');summary['spectator']=r
 return summary

def hashes():
 pins=json.loads((HERE/'source_freeze.json').read_text());bad=[k for k,v in pins.items()if hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=v]
 if bad:raise RuntimeError('Native glyph candidate drift '+str(bad))
 return pins

def run(egl,physical,out):
 from research.v4.accelerated.tabletop import create_scene
 from research.v4.bridge.run_b0 import observe_physical_gpu
 if out.exists():raise RuntimeError('Fresh output required; old evidence cannot be overwritten')
 out.mkdir(parents=True);pins=hashes();started=time.monotonic();s=None;checkpoints=[]
 report=dict(passed=False,error=None,source_sha256=pins,controls=0,added_sim_forward=0,added_state_restore=0,models_called=[],simulator_pid=os.getpid(),render_gpu_device_id=egl,requested_physical_gpu=physical,online_protocol_adopted=False,review_state='pending_root_actual_public_view_review',intentional_visual_change='Only isolated glyph geom alpha; RGB/depth differences are measured, not required equal')
 def expired(*_):raise TimeoutError('Native pair120s')
 prior=signal.signal(signal.SIGALRM,expired);signal.alarm(120)
 try:
  s=create_scene('R2-01',0,camera_observations=True,render_gpu_device_id=egl)
  def forbid(*_args,**_kw):raise RuntimeError('Zero-control native proof forbids step')
  s.step=forbid;s.sim.step=forbid;s.sim.forward=forbid
  if s.sim._render_context_offscreen is None:raise RuntimeError('Existing constructor render context required; no new-context forward')
  m=s.sim.model._model;original=arrays(m);options=arrays(m.opt);scalars={k:getattr(m.opt,k)for k in dir(m.opt)if isinstance(getattr(m.opt,k),(int,float,bool))};camids=[m.camera(v).id for v in VIEWS];camera={k:np.asarray(getattr(m,k)[camids]).copy()for k in('cam_pos','cam_quat','cam_fovy')};base=state(s);initial_time=float(s.sim.data.time)
  first=capture(s);p1=props(s);baseline=capture(s);p2=props(s)
  if not prop_equal(p1,p2)or not exact(base,state(s)):raise RuntimeError('Initial fresh proprio/integration baseline is not stable')
  def checkpoint(name):
   row=dict(name=name,integration_exact=exact(base,state(s)),camera_arrays_exact=all(exact(v,np.asarray(getattr(m,k)[camids]))for k,v in camera.items()),proprio_exact=prop_equal(p2,props(s)),actual_proprio=serialize(props(s)),sim_time=float(s.sim.data.time));checkpoints.append(row)
   if not all(row[k]for k in('integration_exact','camera_arrays_exact','proprio_exact')):raise RuntimeError('Same-state/camera/proprio proof failed '+name)
  checkpoint('fresh_before');sp0=spectator(s);s.raw=s._get_observations(force_update=True);checkpoint('after_before_HD');gpu=observe_physical_gpu(os.getpid(),physical);report['physical_gpu']=gpu
  if not gpu['observed']:raise RuntimeError('Actual physical graphics PID not observed')
  save_packet(out/'before',baseline,sp0)
  patch=helper.hide_glyph(m);g=patch['geom_id'];changed=arrays(m);delta={k:np.argwhere(~(original[k]==changed[k])).tolist()for k in original if not exact(original[k],changed[k])}
  if delta!={'geom_rgba':[[g,3]]}or not all(exact(v,arrays(m.opt)[k])for k,v in options.items())or any(getattr(m.opt,k)!=v for k,v in scalars.items()):raise RuntimeError('Unexpected model/option mutation')
  report['model_proof']=dict(checked_model_arrays=len(original),checked_option_arrays=len(options),option_numeric_scalars_exact=True,exact_changed_elements=delta,patch=patch,mass_inertia_collisions_topology_camera_unchanged=True);checkpoint('alpha_applied')
  after=capture(s);checkpoint('fresh_after');sp1=spectator(s);s.raw=s._get_observations(force_update=True);checkpoint('after_after_HD');save_packet(out/'after',after,sp1)
  report['image_differences']=differences(out/'difference_maps',baseline,after,sp0,sp1)
  if not all(report['image_differences'][v]['K_exact']and report['image_differences'][v]['T_exact']for v in VIEWS):raise RuntimeError('Public calibration changed')
  context=s.sim._render_context_offscreen;report.update(passed=True,initial_sim_time_s=initial_time,final_sim_time_s=float(s.sim.data.time),fresh_baseline_proprio=serialize(p2),render_context=dict(type=type(context).__name__,device_id=getattr(context,'device_id',None),offscreen=True,public_native_resolution=[640,480],spectator_native_resolution=[1920,1080]),checkpoints=checkpoints,source_drift=False,notice_preserved=True)
 except BaseException as exc:report['error']=type(exc).__name__+': '+str(exc)
 finally:
  if s is not None:s.close()
  signal.alarm(0);signal.signal(signal.SIGALRM,prior);report.update(checkpoints=checkpoints,wall_seconds=time.monotonic()-started)
  try:report['source_drift']=pins!=hashes()
  except Exception as exc:report['source_drift']=True;report['source_drift_error']=str(exc)
  (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--egl-device',type=int,choices=(0,1),required=True);p.add_argument('--physical-gpu',type=int,choices=(0,1),required=True);p.add_argument('--out',type=Path,default=HERE/'native_pair');p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 if a.preflight_only:print(json.dumps(dict(source_count=len(hashes()),controls=0,models=[],will_construct_scene=False,public=[640,480],spectator=[1920,1080])));return
 os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(a.egl_device));r=run(a.egl_device,a.physical_gpu,a.out);print(json.dumps(dict(passed=r['passed'],error=r['error'],wall_seconds=r['wall_seconds'],source_drift=r['source_drift'])));raise SystemExit(0 if r['passed']and not r['source_drift']else 2)
if __name__=='__main__':main()
