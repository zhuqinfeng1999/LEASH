"""Actual original agentview pose rendered offline at HD, fixed physical prefix."""
import argparse, hashlib, json, os, signal, time, traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from research.v4.candidates.scene_review_r1_variants_20260926 import export as original

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
INSTANCES=('R1-02','R1-03')

def hashes(instance):
    pins=original.hashes(instance)
    for p in [*HERE.glob('*.py'),HERE/'protocol.json']:
        pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    freeze=HERE/'source_freeze.json'
    if freeze.exists()and pins!=json.loads(freeze.read_text())[instance]:
        raise ValueError('isolated original agentview display source drift')
    return dict(sorted(pins.items()))

class AgentviewHDRecorder(original.measured.MeasuredFreeRecorder):
    def native_frame(self,scene):
        # Only an offline recorder's named camera selection changes. Actual
        # public camera pose/FOV, RGB-D buffers and model are never changed.
        with patch.object(scene,'spectator_view','agentview'):
            return original.recorder.NativeSceneRecorder.native_frame(self,scene)
    def begin_segment(self,scene,**kwargs):
        sim=scene.backend.sim;cid=sim.model.camera_name2id('agentview')
        pose=dict(renderer_camera='agentview used only as offline HD observer',
                  position_world_m=np.asarray(sim.data.cam_xpos[cid]).tolist(),
                  rotation_world=np.asarray(sim.data.cam_xmat[cid]).reshape(3,3).tolist(),
                  fovy_deg=float(sim.model.cam_fovy[cid]),online_protocol_changed=False,
                  display_version='original-public-agentview-pose-native-HD-v2')
        with patch.object(original.observer,'POSE',pose):
            return super().begin_segment(scene,**kwargs)

class PrefixPreviewComplete(Exception):
    """Intentional display-only cancellation at the first actual 20mm lift."""

class PrefixPreviewRecorder(AgentviewHDRecorder):
    def write(self,frame,*,instruction,chapter,line,repeat=1):
        self.frame_count+=repeat
        if self.frame_count==self.fps:
            Image.fromarray(frame).save(self.path.parent/'initial_agentview_HD.png')
        if line.startswith('Actual between-finger attachment'):
            Image.fromarray(frame).save(self.path.parent/'actual_attachment_agentview_HD.png')
        if line.startswith('Actual attachment and at least 2 cm lift'):
            Image.fromarray(frame).save(self.path.parent/'actual_held_lift_agentview_HD.png')
            self.preview_held=True
            raise PrefixPreviewComplete('display-only stop after original first actual 20mm lift')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--instance',choices=INSTANCES,required=True)
    ap.add_argument('--egl-device',type=int,default=0);ap.add_argument('--physical-gpu',type=int,default=0)
    ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args();pins=hashes(a.instance)
    folder=HERE/a.instance/'native_preview';capture=folder/'capture'
    if a.preflight_only:
        print(json.dumps(dict(instance=a.instance,source_count=len(pins),controls_cap=1200,
                              wall_cap_s=180,models_called=[],will_construct_scene=False,
                              output_available=not folder.exists(),camera='original actual agentview')));return
    if folder.exists():raise RuntimeError('fresh preview required; do not overwrite failures')
    capture.mkdir(parents=True);started=time.monotonic();r=PrefixPreviewRecorder(capture/'unused.mp4')
    report=dict(instance_id=a.instance,kind='OFFLINE_HD_PREFIX_PREVIEW_NOT_QUALIFICATION_OR_METHOD',
                source_sha256=pins,models_called=[],simulator_pid=os.getpid(),
                original_canonical_script_unchanged=True,online_protocol_changed=False,
                ordinary_original_control_prefix_only=True,preview_stopped_at_actual_lift=False,error=None)
    def stop(*_):raise TimeoutError('R1 original agentview prefix preview180s')
    old=signal.signal(signal.SIGALRM,stop);signal.alarm(180)
    try:
        os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(a.egl_device))
        from research.v4.bridge.run_b0 import observe_physical_gpu
        original_paths=lambda _: (capture,folder/(a.instance+'.unused.mp4'))
        with patch.object(original,'paths',original_paths):
            result=original.run_case(a.instance,'naive',r,a,report,observe_physical_gpu)
        report.update(prefix_result=result,controls=result['controls'],native_resolution=[r.width,r.height],
                      render_state_checks=r.render_state_checks,chapters=r.segment,
                      preview_stopped_at_actual_lift=bool(getattr(r,'preview_held',False)),
                      partial_prefix_label_not_method_result=result['result']['label'])
        if not report['preview_stopped_at_actual_lift']or not str(result['error']).startswith('PrefixPreviewComplete:'):
            raise RuntimeError('preview did not reach original first actual lift; retain failure')
        if result['controls']>1200 or pins!=hashes(a.instance):raise RuntimeError('prefix cap/source drift')
        report['source_drift']=False
    except BaseException as exc:report.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old);report['wall_seconds']=time.monotonic()-started
        (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report.get(k)for k in ('instance_id','error','controls','preview_stopped_at_actual_lift','native_resolution','wall_seconds')}))
    return 2 if report['error']else 0

if __name__=='__main__':raise SystemExit(main())
