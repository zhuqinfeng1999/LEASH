"""Thin root-only native stages for independently requalified R402/R405."""
import argparse,hashlib,json,os,signal,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from research.v4.candidates.r4_derived_first_risk_20260926 import engine,preflight
from research.v4.candidates.r4_06_raised_first_risk_native_20260926 import native as stage_parent
StageMixin=stage_parent.StageMixin
NativeSceneRecorder=stage_parent.NativeSceneRecorder
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def hashes():
 pins={**stage_parent.hashes(),**preflight.hashes()}
 for p in [*HERE.glob('*.py'),HERE/'protocol.json',HERE/'cpu_contract.json']:
  pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 pins=dict(sorted(pins.items()));f=HERE/'source_freeze.json'
 if f.exists() and pins!=json.loads(f.read_text()):raise RuntimeError('Derived native source drift')
 return pins
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--instance',choices=['R4-02','R4-05'],required=True);ap.add_argument('--egl-device',type=int,default=0);ap.add_argument('--physical-gpu',type=int,default=0);ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args();pins=hashes()
 qualification=json.loads((engine.HERE/a.instance/'qualification30_summary.json').read_text())
 if qualification['source_sha256']!=preflight.hashes() or qualification['script_cases_passed']!=30 or qualification['initial_geometry_passed']!=10:raise RuntimeError('Exact current source qualification required')
 if a.preflight_only:print(json.dumps(dict(instance=a.instance,source_count=len(pins),models_called=[],will_construct_scene=False,controls_cap=1200,wall_cap_s=180)));return 0
 c=engine.Configuration(a.instance);work=HERE/a.instance/'native_seed0'
 if work.exists():raise RuntimeError('Fresh output required, preserve old evidence')
 work.mkdir(parents=True);start=time.monotonic();report=dict(instance_id=a.instance,seed=0,models_called=[],source_sha256=pins,simulator_pid=os.getpid(),case='naive',stages=[],script_result=None,error=None,full_transport_or_release_claim=False)
 def expired(*_):raise TimeoutError('Derived native-stage180s')
 old=signal.signal(signal.SIGALRM,expired);signal.alarm(180);factory=c.Scene;rec=NativeSceneRecorder(work/'unused.mp4',width=640,height=480)
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
    for camera,camera_packet in packet['cameras'].items():
     if camera_packet['rgb'].shape!=(480,640,3) or camera_packet['depth_m'].shape!=(480,640):raise RuntimeError('Actual two-public640RGBD required')
     Image.fromarray(camera_packet['rgb']).save(work/f'{name}_{camera}.png');np.savez_compressed(work/f'{name}_{camera}.npz',**camera_packet);meta[camera]=dict(rgb_shape=list(camera_packet['rgb'].shape),depth_shape=list(camera_packet['depth_m'].shape))
    Image.fromarray(rec.native_frame(self)).save(work/f'{name}_spectator.png');equal=bool(np.array_equal(before,rec._state(self)))
    if not equal:raise RuntimeError('Stage capture changed integration')
    report['stages'].append(dict(name=name,sequence=self.sequence,sim_time_s=float(sim.data.time),held=bool(self.grasp_assist.held),public_cameras=meta,integration_equal=equal,spectator_resolution=[640,480]));self._saved.add(name)
  with patch.dict(c.namespace,{c.scene_token:Observed}):report['script_result']=c.run_case(0,'naive')
  report['source_drift']=pins!=hashes()
  if report['source_drift']:raise RuntimeError('Source drift')
  r=report['script_result']
  if not r['passed'] or r['evaluator']['label']not in('HARM','UNSAFE_COMMIT') or r['release_performed'] or not r['stop_witness']['actual_assist_held'] or r['stop_witness']['actual_lift_m']<.02:raise RuntimeError('Unexpected original danger witness')
 except BaseException as exc:report['error']=f'{type(exc).__name__}: {exc}'
 finally:
  signal.alarm(0);signal.signal(signal.SIGALRM,old);report['wall_seconds']=time.monotonic()-start;(work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(instance=a.instance,error=report['error'],pid=report['simulator_pid'],stages=[x['name']for x in report['stages']],wall_s=report['wall_seconds'])));return 2 if report['error']else 0
if __name__=='__main__':raise SystemExit(main())
