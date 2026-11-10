"""Root-only120s/20-neutral native view QA; never invoked by CPU qualification."""
import argparse,hashlib,json,os,signal,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from research.v4.bridge.run_b0 import observe_physical_gpu
from .offline_evaluator import Evaluator
from .scene import T1R3ThermosScene,HERE
from .qualify_cpu import hashes
from .project_cpu import screen
ROOT=HERE.parents[3]

def save(scene,out,prefix):
    packet=scene.capture_agent()
    if set(packet['cameras'])!={'agentview','robot0_eye_in_hand'}:raise RuntimeError('two public cameras required')
    for name,c in packet['cameras'].items():
        rgb=np.asarray(c['rgb']);depth=np.asarray(c['depth_m'])
        if rgb.shape!=(480,640,3) or depth.shape!=(480,640) or not np.isfinite(depth).all() or not np.all(depth>0):raise RuntimeError('actual640x480RGB-D required')
        Image.fromarray(rgb).save(out/f'{prefix}_{name}.png')
        np.savez_compressed(out/f'{prefix}_{name}_rgbd.npz',rgb=rgb,depth_m=depth,intrinsics=c['intrinsics'],extrinsics=c['extrinsics'])
    Image.fromarray(scene.spectator_rgb()).save(out/f'{prefix}_spectator.png')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--physical-gpu',type=int,choices=(0,1),required=True);p.add_argument('--render-device',type=int,choices=(0,1),required=True);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    out=a.output.resolve()
    if out.parent!=HERE.resolve() or out.exists():raise ValueError('fresh candidate-local output required')
    if os.environ.get('MUJOCO_EGL_DEVICE_ID')!=str(a.render_device):raise ValueError('explicit matching EGL device required')
    before=hashes();out.mkdir();(out/'source-sha256.json').write_text(json.dumps(before,indent=2)+'\n')
    started=time.monotonic();scene=None;error=None
    report={'kind':'R3-03_STEEL_THERMOS_MICROWAVE_NATIVE_VIEW_QA_NOT_METHOD','seed':a.seed,'pid':os.getpid(),'control_steps':0,'source_hashes':before,'native_acceptance':'PENDING_ROOT_REVIEW'}
    old=signal.getsignal(signal.SIGALRM)
    def alarm(*_):raise TimeoutError('nativeQA120s limit')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(120)
    try:
        scene=T1R3ThermosScene(a.seed,camera_observations=True,render_gpu_device_id=a.render_device)
        report['physical_gpu_placement']=observe_physical_gpu(os.getpid(),a.physical_gpu)
        if not report['physical_gpu_placement']['observed']:raise RuntimeError('PID not observed on selected physical GPU')
        evaluator=Evaluator();initial=evaluator.observe(scene)
        if initial['harm'] or initial['unsafe_commit']:raise RuntimeError('initialdanger')
        save(scene,out,'initial');report['initial_offline_screen']=screen(scene.backend.sim)
        for _ in range(20):
            scene.step(np.array([0,0,0,0,0,0,-1],np.float32));report['control_steps']+=1
            state=evaluator.observe(scene)
            if state['harm'] or state['unsafe_commit'] or scene.grasp_assist.held:raise RuntimeError('neutral control unexpectedly crossed danger/attached')
        save(scene,out,'neutral20');report['final_offline_screen']=screen(scene.backend.sim)
        (out/'scene.xml').write_text(scene.backend.model.get_xml())
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';report['traceback']=traceback.format_exc()
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old)
        if scene is not None:scene.close()
        report.update(error=error,elapsed_s=time.monotonic()-started,source_drift=[n for n,h in before.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h])
        report['passed_diagnostic']=error is None and not report['source_drift']
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ['pid','error','elapsed_s','source_drift','passed_diagnostic']}))
    return 0 if report['passed_diagnostic'] else 1
if __name__=='__main__':raise SystemExit(main())
