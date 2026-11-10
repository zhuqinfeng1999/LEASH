"""Bounded offline actual R406 stages; root renderer only, no model calls."""
import argparse,hashlib,json,os,signal,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from research.v4.candidates.r4_06_first_risk_raised3mm_20260926 import scripts,preflight
from research.v4.candidates.scene_review_export_20260926.recorder import NativeSceneRecorder
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def hashes():
 pins=preflight.hashes()
 for p in[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'cpu_contract.json',Path(NativeSceneRecorder.__module__.replace('.','/')).with_suffix('.py')]:
  p=p if p.is_absolute()else ROOT/p;pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 pins=dict(sorted(pins.items()));f=HERE/'source_freeze.json'
 if f.exists()and pins!=json.loads(f.read_text()):raise RuntimeError('R406 native-stage source drift')
 return pins
class StageMixin:
 def _initialize_stage(self):self._saved=set();self._capture_stage('initial')
 def step(self,a):
  r=super().step(a)
  if self.grasp_assist.held and 'actual_attach'not in self._saved:self._capture_stage('actual_attach')
  if self.grasp_assist.held:
   p=self.backend.sim.data.get_joint_qpos(self.backend.objects_dict[self.item_name].joints[0])
   if p[2]-self.initial_item_z>=.02 and 'actual_lift_2cm'not in self._saved:self._capture_stage('actual_lift_2cm')
  return r
 def close(self):
  try:self._capture_stage('final_first_actual_risk')
  finally:super().close()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--egl-device',type=int,default=0);ap.add_argument('--physical-gpu',type=int,default=0);ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args();pins=hashes()
 if a.preflight_only:print(json.dumps(dict(source_count=len(pins),models_called=[],will_construct_scene=False,controls_cap=1200,wall_cap_s=180,scene_source_pins=91)));return
 work=HERE/'native_seed0'
 if work.exists():raise RuntimeError('Fresh output required, preserve old evidence')
 work.mkdir();start=time.monotonic();report=dict(instance_id='R4-06',seed=0,models_called=[],source_sha256=pins,simulator_pid=os.getpid(),case='naive',stages=[],script_result=None,error=None,full_transport_or_release_claim=False)
 def expired(*_):raise TimeoutError('R406 native-stage180s')
 old=signal.signal(signal.SIGALRM,expired);signal.alarm(180);factory=scripts.namespace['T1R4CoinDishScene'];rec=NativeSceneRecorder(work/'unused.mp4',width=640,height=480)
 try:
  os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(a.egl_device))
  from research.v4.bridge.run_b0 import observe_physical_gpu
  class Observed(StageMixin,factory):
   def __init__(self,*args,**kwargs):
    kwargs.update(camera_observations=True,render_gpu_device_id=a.egl_device);super().__init__(*args,**kwargs);report['physical_gpu']=observe_physical_gpu(os.getpid(),a.physical_gpu)
    if not report['physical_gpu']['observed']:factory.close(self);raise RuntimeError('Actual physical GPU/PID unverified')
    try:self._initialize_stage()
    except BaseException:factory.close(self);raise
   def _capture_stage(self,name):
    before=rec._state(self);packet=self.capture_agent();meta={};sim=self.backend.sim
    for camera,c in packet['cameras'].items():
     if c['rgb'].shape!=(480,640,3)or c['depth_m'].shape!=(480,640):raise RuntimeError('Actual two-public640RGBD required')
     Image.fromarray(c['rgb']).save(work/f'{name}_{camera}.png');np.savez_compressed(work/f'{name}_{camera}.npz',**c);meta[camera]=dict(rgb_shape=list(c['rgb'].shape),depth_shape=list(c['depth_m'].shape))
    image=rec.native_frame(self);Image.fromarray(image).save(work/f'{name}_spectator.png');equal=bool(np.array_equal(before,rec._state(self)))
    if not equal:raise RuntimeError('Stage capture changed integration')
    report['stages'].append(dict(name=name,sequence=self.sequence,sim_time_s=float(sim.data.time),held=bool(self.grasp_assist.held),public_cameras=meta,integration_equal=equal,spectator_resolution=[640,480]));self._saved.add(name)
  with patch.dict(scripts.namespace,{'T1R4CoinDishScene':Observed}):report['script_result']=scripts.run_case(0,'naive')
  report['source_drift']=pins!=hashes()
  if report['source_drift']:raise RuntimeError('Source drift')
  if not report['script_result']['passed']or report['script_result']['evaluator']['label']not in('HARM','UNSAFE_COMMIT'):raise RuntimeError('Unexpected original danger witness')
 except BaseException as exc:report['error']=f'{type(exc).__name__}: {exc}'
 finally:
  signal.alarm(0);signal.signal(signal.SIGALRM,old);report['wall_seconds']=time.monotonic()-start;(work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(error=report['error'],pid=report['simulator_pid'],stages=[x['name']for x in report['stages']],wall_s=report['wall_seconds'])));return 2 if report['error']else 0
if __name__=='__main__':raise SystemExit(main())
