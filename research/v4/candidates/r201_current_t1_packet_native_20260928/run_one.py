"""Root-only actual current R201 packet; no main, task motion or model I/O."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from PIL import Image

from research.v4.candidates.r201_current_t1_method_binding_20260928 import binding
from research.v4.candidates.panda_glyph_native_pair_20260927 import native_pair as packet
from research.v4.candidates.panda_local_cutout_native_20260927 import native_pair as guard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def admit():
    pins = json.loads((HERE / 'source_freeze.json').read_text())
    for name, digest in pins.items():
        p = Path(name) if name.startswith('/') else ROOT / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise RuntimeError('R201 packet source drift: ' + name)
    return pins


def run(device, out):
    if out.exists():
        raise ValueError('Fresh bounded output required')
    pins = admit()
    out.mkdir(parents=True)
    s = None
    started = time.monotonic()
    report = dict(passed=False, error=None, source_sha256=pins,
                  simulator_pid=os.getpid(), requested_physical_gpu=device,
                  controls=0, extra_forward_or_restore=0, models_services_main=0,
                  capture_kind='actual constructor cached two-public-frame packet',
                  natural_method=False, counts_added=0)
    checks = []
    def expire(*_):
        raise TimeoutError('R201 packet 120s wall bound')
    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(120)
    try:
        s = binding.CurrentR201Scene(0, render_gpu_device_id=device)
        report['constructor_completed_seconds'] = time.monotonic() - started
        b = s.backend
        raw_before = {k:np.asarray(v).copy() for k,v in s.raw.items() if isinstance(v,np.ndarray)}
        base = guard.immutable_snapshot(b)
        sequence = s.sequence
        s.step = b.step = b.sim.step = b.sim.forward = guard.forbid
        public = s.capture_agent()
        if set(public['cameras']) != set(packet.VIEWS):
            raise RuntimeError('Exactly two public cameras required')
        if set(public['robot_state']) != {'eef_pos','eef_quat','gripper_qpos','grasp_latched'}:
            raise RuntimeError('Only common proprio and declared boolean latch allowed')
        if public['robot_state']['grasp_latched'] is not False:
            raise RuntimeError('Unheld initial latch expected')
        from robosuite.utils.camera_utils import get_real_depth_map, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
        guards = {}
        for view in packet.VIEWS:
            c = public['cameras'][view]
            guards[view] = dict(
                rgb_exact=packet.exact(c['rgb'],np.ascontiguousarray(raw_before[view+'_image'][::-1])),
                metric_depth_exact=packet.exact(c['depth_m'],np.ascontiguousarray(get_real_depth_map(b.sim,raw_before[view+'_depth'].squeeze())[::-1])),
                K_exact=packet.exact(c['intrinsics'],get_camera_intrinsic_matrix(b.sim,view,480,640)),
                T_exact=packet.exact(c['extrinsics'],get_camera_extrinsic_matrix(b.sim,view)),
                actual_rgb_shape=list(c['rgb'].shape), actual_depth_shape=list(c['depth_m'].shape),
                same_sequence=c['frame_id']==sequence)
            if not all(guards[view][k] for k in ('rgb_exact','metric_depth_exact','K_exact','T_exact','same_sequence')):
                raise RuntimeError('Actual constructor sensor-packet guard '+view)
        for field,key in zip(packet.FIELDS,('eef_pos','eef_quat','gripper_qpos')):
            if not packet.exact(public['robot_state'][key],raw_before[field]):
                raise RuntimeError('Actual cached public proprio differs')
        guard.same_physics(base,guard.immutable_snapshot(b))
        with guard.preserved_camera(b,checks,'existing_named_offline_spectator1080'):
            spec=np.ascontiguousarray(b.sim.render(width=1920,height=1080,camera_name=s.spectator_view)[::-1])
        if spec.shape!=(1080,1920,3) or spec.dtype!=np.uint8:
            raise RuntimeError('Native1080 spectator required')
        again=s.capture_agent()
        for view in packet.VIEWS:
            if any(not packet.exact(public['cameras'][view][k],again['cameras'][view][k]) for k in ('rgb','depth_m','intrinsics','extrinsics')):
                raise RuntimeError('Cached packet mutated by offline rendering')
        report['physical_guard']=guard.same_physics(base,guard.immutable_snapshot(b))
        report['constructor_sensor_guards']=guards
        report['sequence_unchanged']=s.sequence==sequence
        from research.v4.bridge.run_b0 import observe_physical_gpu
        report['physical_gpu']=observe_physical_gpu(os.getpid(),device)
        if not report['physical_gpu']['observed']:
            raise RuntimeError('Actual physical graphics PID not observed')
        where=out/'actual'
        where.mkdir()
        for view,c in public['cameras'].items():
            Image.fromarray(c['rgb']).save(where/(view+'.png'))
            np.savez_compressed(where/(view+'_RGBD_K_T.npz'),**{k:c[k] for k in ('rgb','depth_m','intrinsics','extrinsics')})
        Image.fromarray(spec).save(where/'offline_spectator_1080.png')
        report.update(passed=True, public_proprio={k:np.asarray(v).tolist() for k,v in public['robot_state'].items()},
                      postconstructor_task_controls=0, initial_sequence=sequence,
                      initial_sim_time_s=float(b.sim.data.time), cached_not_fresh_recapture=True)
    except BaseException as e:
        report['error']=type(e).__name__+': '+str(e)
    finally:
        if s is not None:
            s.close()
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous)
        report.update(wall_seconds=time.monotonic()-started, context_checks=checks)
        try:
            report['source_drift']=pins!=admit()
        except Exception as e:
            report.update(source_drift=True,source_error=str(e))
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',type=int,choices=(0,1),required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    if a.preflight_only:
        print(json.dumps(dict(source_count=len(admit()),constructors=0,controls=0,models=0)))
        return
    os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(a.device))
    r=run(a.device,a.out)
    print(json.dumps({k:r[k] for k in ('passed','error','wall_seconds','source_drift')}))
    raise SystemExit(0 if r['passed'] and not r['source_drift'] else 2)


if __name__=='__main__':
    main()
