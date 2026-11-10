"""Root-only finite actual naive motion capture; all observations are offline."""
import argparse,hashlib,json,os,signal,time,traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import scripts
from .qualify import HERE,ROOT
from research.v4.candidates.r4_05_inclined_identity_v2_20260926 import native_qa as original
from research.v4.candidates.scene_review_export_20260926 import recorder as recording
from research.v4.bridge.run_b0 import observe_physical_gpu

def hashes():
    pins=json.loads((HERE/'source_freeze.json').read_text())
    pins[str(Path(recording.__file__).relative_to(ROOT))]=hashlib.sha256(Path(recording.__file__).read_bytes()).hexdigest()
    for name,sha in pins.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('native motion source drift: '+name)
    frozen=HERE/'native_source_freeze.json'
    if frozen.exists()and pins!=json.loads(frozen.read_text()):raise ValueError('native union drift')
    return pins

def run_capture(*,output,seed=0,render_device=0,physical_gpu=0,render=True):
    start=time.monotonic();pins=hashes();output=Path(output);records=[];rec=recording.NativeSceneRecorder(output/'unused.mp4',width=640,height=480);control_count=0
    report=dict(kind='R4-05_REAL_SCRIPTED_MOTION_OFFLINE_NOT_METHOD',seed=seed,instance_id='R4-05',instruction='Put the item on the left into the crib.',pid=os.getpid(),rendering_enabled=render,models_called=[],public_views=['agentview','robot0_eye_in_hand'],public_rgbd_resolution=[640,480],spectator_online=False,source_sha256=pins,control_cap=1200,wall_cap_s=180,stages=records,error=None)
    scene_type=scripts.namespace['T1R4MagneticBoxScene']
    class CaptureScene(scene_type):
        def __init__(self,*args,**kwargs):
            kwargs.update(camera_observations=render,render_gpu_device_id=render_device if render else -1)
            super().__init__(*args,**kwargs);self._capture_closed=False;self._capture_lifted=False;self._capture_release=False;self._positive_held=0
            if render:
                report['physical_gpu_placement']=observe_physical_gpu(os.getpid(),physical_gpu)
                if not report['physical_gpu_placement']['observed']:super().close();raise RuntimeError('physical GPU PID unverified before motion')
            try:snapshot(self,'initial')
            except BaseException:super().close();raise
        def step(self,action):
            nonlocal control_count
            if control_count>=1200 or time.monotonic()-start>180:raise TimeoutError('native motion1200controls/180s')
            if float(action[6])<0 and self.grasp_assist.held and not self._capture_release:
                self._capture_release=True;snapshot(self,'before_release')
            result=super().step(action);control_count+=1
            if self.grasp_assist.held and float(action[6])>0:
                self._positive_held+=1
                if self._positive_held==1:snapshot(self,'actual_attachment')
                if self._positive_held==24:
                    self._capture_closed=True;snapshot(self,'closed_actual')
            item=self.backend.objects_dict[self.item_name]
            height=float(self.backend.sim.data.get_joint_qpos(item.joints[0])[2]-self.initial_item_z)
            if self.grasp_assist.held and height>=.02 and not self._capture_lifted:
                self._capture_lifted=True;snapshot(self,'lifted_2cm')
            return result
        def close(self):
            try:snapshot(self,'final_after_script')
            finally:super().close()
    def snapshot(scene,name):
        sim=scene.backend.sim;m=sim.model._model;ids=[sim.model.camera_name2id(n)for n in scene.public_views];before=rec._state(scene);cams={k:np.asarray(getattr(m,k)[ids]).copy()for k in ('cam_pos','cam_quat','cam_fovy')}
        row=dict(name=name,control_count=control_count,sim_time_s=float(sim.data.time),attached=bool(scene.grasp_assist.held),offline_capture_only=True)
        if render:
            with patch.object(scene,'spectator_rgb',lambda:rec.native_frame(scene)):original.save(scene,output,name)
            row['files']=[str(output/f'{name}_{v}.png')for v in ['agentview','robot0_eye_in_hand','spectator']]
        row.update(integration_state_equal=bool(np.array_equal(before,rec._state(scene))),public_camera_model_equal=all(np.array_equal(v,np.asarray(getattr(m,k)[ids]))for k,v in cams.items()))
        if not row['integration_state_equal']or not row['public_camera_model_equal']:raise RuntimeError('offline capture changed physical state or public cameras')
        records.append(row)
    try:
        with patch.dict(scripts.namespace,{'T1R4MagneticBoxScene':CaptureScene}):result=scripts.run_case(seed,'naive')
        report.update(result=result,control_steps=control_count)
        if not result['passed']or result['error']or result['control_steps']!=control_count:raise RuntimeError('actual script qualification mismatch')
        if [r['name']for r in records]!=['initial','actual_attachment','closed_actual','lifted_2cm','before_release','final_after_script']:raise RuntimeError('required actual motion stages missing')
    except BaseException as exc:report.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    report.update(control_steps=control_count,elapsed_s=time.monotonic()-start,source_drift=pins!=hashes(),native_render_state_checks=rec.render_state_checks,passed=not report['error'],native_review='pending_root'if render else'CPU_BINDING_ONLY')
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--render-device',type=int,choices=(0,1),required=True);p.add_argument('--physical-gpu',type=int,choices=(0,1),required=True);a=p.parse_args();out=a.output.resolve()
    if out.parent!=HERE or out.exists():raise ValueError('fresh candidate-local output required')
    if os.environ.get('MUJOCO_EGL_DEVICE_ID')!=str(a.render_device):raise ValueError('explicit matching EGL device required')
    out.mkdir();old=signal.getsignal(signal.SIGALRM)
    def stop(*_):raise TimeoutError('native motion180s wall')
    signal.signal(signal.SIGALRM,stop);signal.alarm(180)
    try:r=run_capture(output=out,seed=a.seed,render_device=a.render_device,physical_gpu=a.physical_gpu)
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k]for k in ['error','passed','control_steps','elapsed_s','source_drift']}))
    return 0 if r['passed'] else 2
if __name__=='__main__':raise SystemExit(main())
