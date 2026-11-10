"""Bounded same-state native RGB refresh evidence; never enlarges any guard."""
import argparse,hashlib,json,os,signal,time
from pathlib import Path
import numpy as np
from research.v4.candidates.scene_review_r1_variants_20260926 import export as hd
from research.v4.candidates.scene_review_r3_02_render_diag_20260926.diagnose import compare
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PRIOR=hd.HERE/'R1-04/capture/chapter1_camera_guard.json'
def hashes():
    pins=hd.hashes('R1-04')
    for p in [*HERE.glob('*.py'),HERE/'protocol.json',PRIOR,PRIOR.parent/'report.json']:
        pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    pins=dict(sorted(pins.items()));f=HERE/'source_freeze.json'
    if f.exists()and pins!=json.loads(f.read_text()):raise ValueError('R104 readonly diagnostic source drift')
    return pins
def sequence(scene,r,check,rows):
    def force():scene.raw=scene.backend._get_observations(force_update=True);return scene.capture_agent()
    for block,observer in [('plain_before',False),('observer_interleaved',True),('plain_after',False)]:
        for i in range(6):
            a=force();b=scene.capture_agent();check();cached=compare(a,b)
            if observer:r.native_frame(scene);check()
            after_observer=scene.capture_agent();check();c=force();check()
            pixels={}
            for name in scene.public_views:
                x=b['cameras'][name]['rgb'];y=c['cameras'][name]['rgb'];delta=x.astype(np.int16)-y.astype(np.int16);indices=np.argwhere(delta!=0)
                pixels[name]=dict(changed_pixel_count=int(np.count_nonzero(np.any(delta,axis=2))),changed_channels=len(indices),
                    coordinate_examples_y_x_channel=indices[:40].tolist(),signed_delta_counts={str(int(v)):int(n)for v,n in zip(*np.unique(delta[delta!=0],return_counts=True))},
                    before_rgb_sha256=hashlib.sha256(x.tobytes()).hexdigest(),after_rgb_sha256=hashlib.sha256(y.tobytes()).hexdigest())
            rows.append(dict(block=block,index=i,spectator_rendered=observer,cached_repeat=cached,
                             cached_after_observer=compare(b,after_observer),forced_pair=compare(b,c),changed_pixels=pixels))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--egl-device',type=int,default=0);ap.add_argument('--physical-gpu',type=int,default=0);ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args();pins=hashes();folder=HERE/'native_diagnostic'
    if a.preflight_only:
        print(json.dumps(dict(source_count=len(pins),controls=0,models_called=[],wall_cap_s=120,blocks=['6 plain','6 observer interleaved','6 plain after'],guard_not_changed=True,will_construct_scene=False,output_available=not folder.exists())));return
    if folder.exists():raise RuntimeError('fresh diagnostic required; failures stay immutable')
    folder.mkdir();start=time.monotonic();scene=None
    report=dict(instance_id='R1-04',kind='READONLY_NATIVE_RENDER_DIAGNOSTIC_NOT_QUALIFICATION',controls=0,models_called=[],
                simulator_pid=os.getpid(),source_sha256=pins,error=None,rows=[],exact_checks=[],existing_guard_not_changed=True,
                prior_failed_guard_path=str(PRIOR),prior_failed_guard=json.loads(PRIOR.read_text()))
    def stop(*_):raise TimeoutError('R104 zero-control diagnostic120s')
    old=signal.signal(signal.SIGALRM,stop);signal.alarm(120)
    try:
        os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(a.egl_device))
        from research.v4.bridge.run_b0 import observe_physical_gpu
        scene=hd.provider('R1-04')[2](0,camera_observations=True,render_gpu_device_id=a.egl_device)
        r=hd.observer.FreeObserverRecorder(folder/'unused.mp4');report['gpu']=observe_physical_gpu(os.getpid(),a.physical_gpu)
        if not report['gpu']['observed']:raise RuntimeError('actual physicalGPU PID unverified')
        sim=scene.backend.sim;m,d=sim.model._model,sim.data._data;state=r._state(scene)
        ids=[sim.model.camera_name2id(n)for n in scene.public_views]
        camera={k:np.asarray(getattr(m,k)[ids]).copy()for k in('cam_pos','cam_quat','cam_fovy')}
        proprio={k:np.asarray(scene.raw[k]).copy()for k in('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')}
        ctrl=d.ctrl.copy();time0=float(d.time)
        def check():
            row=dict(integration=bool(np.array_equal(state,r._state(scene))),
                     public_camera_model=all(np.array_equal(v,np.asarray(getattr(m,k)[ids]))for k,v in camera.items()),
                     proprioception=all(np.array_equal(v,np.asarray(scene.raw[k]))for k,v in proprio.items()),
                     ctrl=bool(np.array_equal(ctrl,d.ctrl)),time=float(d.time)==time0)
            report['exact_checks'].append(row)
            if not all(row.values()):raise RuntimeError('readonly render altered physical/public camera state')
        check();sequence(scene,r,check,report['rows']);report['render_state_checks']=r.render_state_checks
        report['native_resolution']=[r.width,r.height]
        report['summary']={block:{name:dict(max_changed_channels=max(x['forced_pair'][name]['rgb']['changed_elements']for x in report['rows']if x['block']==block),max_abs_delta=max(x['forced_pair'][name]['rgb']['max_abs_delta']for x in report['rows']if x['block']==block),non_rgb_exact=all(all(x['forced_pair'][name][k]['equal']for k in('depth_m','intrinsics','extrinsics'))for x in report['rows']if x['block']==block))for name in scene.public_views}for block in('plain_before','observer_interleaved','plain_after')}
        report['source_drift']=pins!=hashes()
        if report['source_drift']:raise RuntimeError('source drift')
    except BaseException as exc:report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if scene is not None:scene.close()
        signal.alarm(0);signal.signal(signal.SIGALRM,old);report['wall_seconds']=time.monotonic()-start
        (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report.get(k)for k in('error','controls','simulator_pid','source_drift','summary','wall_seconds')}))
    return 2 if report['error']else 0
if __name__=='__main__':raise SystemExit(main())
