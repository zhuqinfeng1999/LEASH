"""Bounded existing P1 scripts→native scene-review video; no models or IDs."""
import argparse,hashlib,json,os,signal,time,traceback
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];FINAL=ROOT/'research/v4/review_videos/P1-02.mp4'
def hashes():
    from research.v4.candidates.p1_02_cleaver_t2_20260926.preflight import hashes as base_hashes
    from .recorder import FONT,FONT_NOTICE
    pins=base_hashes()
    for p in [*HERE.glob('*.py'),HERE/'protocol.json',FONT,FONT_NOTICE]:pins[str(p.relative_to(ROOT))if ROOT in p.parents else str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    frozen=HERE/'source_freeze.json'
    if frozen.exists()and pins!=json.loads(frozen.read_text()):raise ValueError('review exporter source drift')
    return pins

def main():
    p=argparse.ArgumentParser();p.add_argument('--egl-device',type=int,default=0);p.add_argument('--physical-gpu',type=int,default=0);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();source=hashes()
    if a.preflight_only:print(json.dumps(dict(instance='P1-02',source_count=len(source),model_calls=False,requested_resolution=[1920,1080],public_resolution=[640,480],output_available=not FINAL.exists())));return
    if FINAL.exists()or FINAL.with_suffix('.json').exists()or(HERE/'capture').exists():raise ValueError('preserve existing final/capture; no retry overwrite')
    os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(a.egl_device))
    from research.v4.candidates.p1_02_cleaver_t2_20260926.scene import P1CleaverFacade,INSTRUCTION
    from research.v4.candidates.p1_clearance_v6_20260925 import candidate_v6 as scripts
    from research.v4.candidates.p1_ma_fallback_20260925 import candidate as base_script
    from research.v4.bridge.run_b0 import observe_physical_gpu
    from .recorder import NativeSceneRecorder,FONT,FONT_NOTICE
    work=HERE/'capture';work.mkdir();video=work/'P1-02.partial.mp4';recorder=NativeSceneRecorder(video);started=time.monotonic();report=dict(instance_id='P1-02',instruction=INSTRUCTION,kind='scripted_high_resolution_scene_inspection_not_method',models_called=[],vla_model_called=False,numbered_episode=False,seed=0,source_sha256=source,source_code_version='p1-02-common-t2-cleaver-tray-v1+native-review-video-v1',physical_gpu_expected=a.physical_gpu,simulator_pid=os.getpid(),public_views=['agentview','robot0_eye_in_hand'],public_rgbd_resolution=[640,480],vla_protocol_changed=False,spectator_online=False,requested_native_resolution=[1920,1080],fps=20,replay_speed=2,sample_stride_controls=2,wall_cap_s=600,control_cap_per_script=1700,cases=[],error=None,review_state='captured_pending_root_review')
    def limit(*_):raise TimeoutError('scene-review export600s wall cap')
    previous=signal.signal(signal.SIGALRM,limit);signal.alarm(600)
    try:
        for mode in('safe','naive'):
            def create(*args,**kwargs):
                kwargs.update(camera_observations=True,render_gpu_device_id=a.egl_device);scene=P1CleaverFacade(*args,**kwargs)
                gpu=observe_physical_gpu(os.getpid(),a.physical_gpu)
                if not gpu['observed']:scene.close();raise RuntimeError('physical GPU PID unverified')
                report.setdefault('physical_gpu_observations',[]).append(gpu);chapter=('Safe route: detour around the arm'if mode=='safe'else'Straight-route hazard inspection: deliberately unsafe script')
                recorder.begin_segment(scene,instruction='P1-02 | '+INSTRUCTION,chapter=chapter);return scene
            with patch.object(scripts,'P1V6Facade',create),patch.object(base_script,'MAX_WALL_S',500):result=scripts.probe(mode,0)
            report['cases'].append(result)
            print(json.dumps(dict(mode=mode,label=(result.get('result')or{}).get('label'),accepted=result['accepted'],script_seconds=result['elapsed_s'],video_frames=recorder.frame_count)),flush=True)
            if not result['accepted']:raise RuntimeError('script physical outcome did not match qualified case; retain failure')
        recorder.close();report.update(source_drift=source!=hashes(),native_resolution=[recorder.width,recorder.height],render_seconds=recorder.render_seconds,encode_write_seconds=recorder.encode_write_seconds,native_captures=recorder.captures,frame_count=recorder.frame_count,video_duration_s=recorder.frame_count/20,chapters=recorder.segment,render_state_checks=recorder.render_state_checks,resolution_fallback_attempts=recorder.resolution_attempts,font=dict(path=str(FONT),license_notice=str(FONT_NOTICE),license='SIL-OFL-1.1',glyphs_rasterized_no_font_redistribution=True))
        if report['source_drift']:raise RuntimeError('source drift')
        import subprocess
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],text=True));stream=next(s for s in probe['streams']if s['codec_type']=='video')
        if(stream['width'],stream['height'])!=(recorder.width,recorder.height)or stream['codec_name']!='h264':raise RuntimeError('encoded format mismatch')
        report['ffprobe']=dict(width=stream['width'],height=stream['height'],codec=stream['codec_name'],frame_rate=stream['r_frame_rate'],duration_s=float(probe['format']['duration']));FINAL.parent.mkdir(exist_ok=True);video.rename(FINAL);report['video_path']=str(FINAL);report['video_sha256']=hashlib.sha256(FINAL.read_bytes()).hexdigest();report['video_bytes']=FINAL.stat().st_size
    except BaseException as exc:report.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),review_state='capture_failed');print(report['error'],flush=True)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        if recorder.pipe is not None and recorder.pipe.poll()is None:
            try:recorder.close()
            except Exception as exc:report['cleanup_error']=f'{type(exc).__name__}: {exc}'
        report['wall_seconds']=time.monotonic()-started;(work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if not report['error']:FINAL.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['error']:raise SystemExit(2)
if __name__=='__main__':main()
