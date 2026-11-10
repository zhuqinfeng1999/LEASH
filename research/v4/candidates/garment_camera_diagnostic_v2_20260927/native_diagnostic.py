"""Single accepted free-camera convention correction, with save-before-fail instrumentation."""
import argparse, hashlib, json, os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import mujoco
import numpy as np
from PIL import Image
from research.v4.candidates.garment_plain_cotton_material_20260927 import native_pair as original

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ORIGINAL_PROOF = original.proof

def hashes():
    pins = dict(ORIGINAL_PROOF.hashes())
    paths = [*HERE.glob('*.py'), HERE/'CAUSE.md', HERE/'HANDOFF.md', HERE/'cpu_report.json', HERE/'fixture_only_cpu_report.json',
             ORIGINAL_PROOF.HERE/'native_pair_gpu1/report.json',
             ROOT/'research/v4/candidates/panda_glyph_close_native_20260927/derive.py']
    for p in paths:
        pins[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    pins = dict(sorted(pins.items()))
    f = HERE/'source_freeze.json'
    if f.exists() and json.loads(f.read_text()) != pins:
        raise RuntimeError('Diagnostic source drift')
    return pins

def configure(c, pose, original_formula=False):
    eye = np.asarray(pose['eye']); target = np.asarray(pose['target']); delta = eye-target
    c.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.cam.fixedcamid = -1; c.cam.trackbodyid = -1; c.cam.lookat[:] = target
    c.cam.distance = float(np.linalg.norm(delta))
    # Exact established MuJoCo FREE-camera convention; unchanged eye/target/elevation.
    offset = 0 if original_formula else 180
    c.cam.azimuth = float((np.degrees(np.arctan2(delta[1], delta[0]))+offset)%360)
    c.cam.elevation = float(-np.degrees(np.arctan2(delta[2], np.linalg.norm(delta[:2]))))

def measured_row(c, frame, name, pose):
    eye = np.asarray(pose['eye']); target = np.asarray(pose['target']); delta = eye-target
    actual_eye = np.mean([cam.pos for cam in c.scn.camera], axis=0)
    actual_forward = np.mean([cam.forward for cam in c.scn.camera], axis=0)
    e = float(np.linalg.norm(actual_eye-eye))
    f = float(np.linalg.norm(actual_forward+delta/np.linalg.norm(delta)))
    return {'name': name+'_actual_pose', 'requested_eye': eye.tolist(), 'requested_target': target.tolist(),
            'actual_eye': actual_eye.tolist(), 'actual_forward': actual_forward.tolist(),
            'eye_error_m': e, 'forward_error': f, 'azimuth_deg': float(c.cam.azimuth),
            'elevation_deg': float(c.cam.elevation), 'frame_shape': list(frame.shape),
            'frame_dtype': str(frame.dtype), 'shape_ok': frame.shape == (1080,1920,3),
            'pose_ok': e <= 1e-5 and f <= 1e-6}

def save_before_check(frame, row, out):
    out.mkdir(parents=True, exist_ok=False)
    frame_path = out/'frame_before_pose_check.png'
    Image.fromarray(frame).save(frame_path)
    row = dict(row, frame_path=str(frame_path), frame_sha256=hashlib.sha256(frame_path.read_bytes()).hexdigest())
    (out/'actual_pose_before_check.json').write_text(json.dumps(row, indent=2)+'\n')
    if not row['shape_ok'] or not row['pose_ok']:
        raise RuntimeError('Actual fixed offline appearance pose mismatch')
    return row

def image(s, checks, name, pose, out):
    with original.guard.preserved_camera(s, checks, name) as c:
        configure(c, pose)
        s.sim.model.vis.global_.fovy = pose['fovy']
        frame = np.ascontiguousarray(s.sim.render(width=1920,height=1080,camera_name=None)[::-1])
        row = measured_row(c, frame, name, pose)
        # Preserve measured values in outer failure report and PNG/JSON before raising.
        checks.append(row)
        save_before_check(frame, row, out/'pose_diagnostics'/name)
        return frame

def run(egl, physical, out):
    proxy = SimpleNamespace(HERE=ORIGINAL_PROOF.HERE, hashes=hashes, initialized=ORIGINAL_PROOF.initialized)
    def instrumented(s, checks, name, pose): return image(s, checks, name, pose, out)
    with patch.object(original, 'proof', proxy), patch.object(original, 'image', instrumented):
        return original.run(egl, physical, out)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--egl-device',type=int,choices=(0,1),required=True)
    p.add_argument('--physical-gpu',type=int,choices=(0,1),required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true'); a=p.parse_args()
    if a.preflight_only:
        print(json.dumps({'source_count':len(hashes()),'controls':0,'model_calls':0,'will_construct':False})); return
    os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(a.egl_device))
    r=run(a.egl_device,a.physical_gpu,a.out)
    print(json.dumps({k:r[k]for k in('passed','error','wall_seconds','source_drift')}))
    raise SystemExit(0 if r['passed'] and not r['source_drift'] else 2)

if __name__=='__main__': main()
