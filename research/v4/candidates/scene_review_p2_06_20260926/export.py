"""One accepted letter-opener script through the existing native review recorder."""
import argparse,hashlib,json,os,signal,time,traceback,subprocess
from pathlib import Path
from unittest.mock import patch
from research.v4.candidates.p2_06_letter_opener_handoff_20260926 import scripts,qualify
from research.v4.candidates.scene_review_free_observer_20260926 import export_p1_02 as observer
from research.v4.candidates.scene_review_export_20260926 import recorder
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
FINAL=ROOT/'research/v4/review_videos/P2-06.mp4'
def hashes():
 pins=qualify.hashes()
 for p in [*HERE.glob('*.py'),HERE/'protocol.json',Path(observer.__file__),Path(recorder.__file__),ROOT/'envs/libero/lib/python3.11/site-packages/robosuite/utils/binding_utils.py',recorder.FONT,recorder.FONT_NOTICE]:
  pins[str(p.relative_to(ROOT))if ROOT in p.parents else str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 freeze=HERE/'source_freeze.json'
 if freeze.exists()and pins!=json.loads(freeze.read_text()):raise ValueError('source drift')
 return pins
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--egl-device',type=int,default=0);ap.add_argument('--physical-gpu',type=int,default=0);ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args();pins=hashes()
 if a.preflight_only:print(json.dumps(dict(instance='P2-06',source_count=len(pins),output_available=not FINAL.exists(),models_called=[])));return
 if FINAL.exists()or FINAL.with_suffix('.json').exists()or(HERE/'capture').exists():raise ValueError('fresh output required; preserve prior capture')
 os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(a.egl_device))
 from research.v4.bridge.run_b0 import observe_physical_gpu
 work=HERE/'capture';work.mkdir();video=work/'P2-06.partial.mp4';r=observer.FreeObserverRecorder(video);start=time.monotonic()
 report=dict(instance_id='P2-06',instruction=scripts.INSTRUCTION,kind='scripted_high_resolution_scene_inspection_not_method',models_called=[],vla_model_called=False,numbered_episode=False,seed=0,source_sha256=pins,source_code_version='p206-current-T3+existing-free-native-observer-v1',physical_gpu_expected=a.physical_gpu,simulator_pid=os.getpid(),public_views=['agentview','robot0_eye_in_hand'],public_rgbd_resolution=[640,480],vla_protocol_changed=False,spectator_online=False,requested_native_resolution=[1920,1080],fps=20,replay_speed=2,sample_stride_controls=2,wall_cap_s=600,control_cap_per_script=1700,cases=[],error=None,review_state='captured_pending_root_review')
 def stop(*_):raise TimeoutError('P206 native review600s wall')
 old=signal.signal(signal.SIGALRM,stop);signal.alarm(600);actual=scripts.P2Facade
 try:
  for mode in ['safe','naive']:
   def create(*args,**kwargs):
    kwargs.update(camera_observations=True,render_gpu_device_id=a.egl_device);scene=actual(*args,**kwargs);gpu=observe_physical_gpu(os.getpid(),a.physical_gpu)
    if not gpu['observed']:scene.close();raise RuntimeError('physicalGPU PID unverified')
    report.setdefault('physical_gpu_observations',[]).append(gpu)
    chapter='Correct scripted placement: handle toward the person'if mode=='safe'else'Deliberately unsafe script: inspect the original hazard predicate'
    with patch.object(observer,'HERE',HERE):r.begin_segment(scene,instruction='P2-06 | '+scripts.INSTRUCTION,chapter=chapter)
    return scene
   with patch.object(scripts,'P2Facade',create),patch.object(scripts,'MAX_WALL_S',500):result=scripts.probe(mode,0)
   report['cases'].append(result);print(json.dumps(dict(mode=mode,accepted=result['accepted'],label=result['result']['label'],steps=result['steps'],frames=r.frame_count)),flush=True)
   if not result['accepted']or result['steps']>1700:raise RuntimeError('script qualification mismatch; retained')
  r.close();report.update(source_drift=pins!=hashes(),native_resolution=[r.width,r.height],render_seconds=r.render_seconds,encode_write_seconds=r.encode_write_seconds,native_captures=r.captures,frame_count=r.frame_count,video_duration_s=r.frame_count/20,chapters=r.segment,render_state_checks=r.render_state_checks,resolution_fallback_attempts=r.resolution_attempts,font=dict(path=str(recorder.FONT),notice=str(recorder.FONT_NOTICE),license='SIL-OFL-1.1'))
  if report['source_drift']:raise RuntimeError('source drift')
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],text=True));s=next(x for x in probe['streams']if x['codec_type']=='video')
  if(s['width'],s['height'])!=(r.width,r.height)or s['codec_name']!='h264':raise RuntimeError('encoded mismatch')
  report['ffprobe']=dict(width=s['width'],height=s['height'],codec=s['codec_name'],frame_rate=s['r_frame_rate'],duration_s=float(probe['format']['duration']));video.rename(FINAL);report.update(video_path=str(FINAL),video_sha256=hashlib.sha256(FINAL.read_bytes()).hexdigest(),video_bytes=FINAL.stat().st_size)
 except BaseException as e:report.update(error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),review_state='capture_failed');print(report['error'],flush=True)
 finally:
  signal.alarm(0);signal.signal(signal.SIGALRM,old)
  if r.pipe is not None and r.pipe.poll()is None:
   try:r.close()
   except Exception as e:report['cleanup_error']=repr(e)
  report['wall_seconds']=time.monotonic()-start;(work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
  if not report['error']:FINAL.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
 if report['error']:raise SystemExit(2)
if __name__=='__main__':main()
