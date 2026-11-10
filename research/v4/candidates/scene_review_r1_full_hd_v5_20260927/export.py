"""Thin approved display-only recorder selection; no moving sensor refresh."""
import argparse,hashlib,json,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.candidates.scene_review_r1_variants_20260926 import export as original
from research.v4.candidates.scene_review_r1_lateral_observer_v3_20260926 import preview as lateral
from research.v4.candidates.scene_review_r1_04_render_diag_20260926 import diagnose
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
ORIGINAL_HASHES=original.hashes;ORIGINAL_RECORDER=original.measured.MeasuredFreeRecorder
DIAGNOSTIC=diagnose.HERE/'native_diagnostic/report.json'
APPROVED_POSE_REPORT=lateral.HERE/'R1-02/native_preview/report.json'
def paths(instance):return HERE/instance/'capture',ROOT/'research/v4/review_videos'/(instance+'.mp4')
def hashes(instance):
    with patch.object(original,'hashes',ORIGINAL_HASHES):
        pins=lateral.hashes(instance)if instance in lateral.prior.INSTANCES else diagnose.hashes()
    for p in [*HERE.glob('*.py'),HERE/'protocol.json',DIAGNOSTIC,APPROVED_POSE_REPORT]:pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    pins=dict(sorted(pins.items()));f=HERE/'source_freeze.json'
    if f.exists()and pins!=json.loads(f.read_text())[instance]:raise ValueError('thin R1 fullHD-v5 source drift')
    return pins
class PublicModelGuardRecorder(ORIGINAL_RECORDER):
    def render_native(self,scene):return original.observer.FreeObserverRecorder.native_frame(self,scene)
    def native_frame(self,scene):
        sim=scene.backend.sim;m=sim.model._model;ids=[sim.model.camera_name2id(n)for n in scene.public_views]
        before={k:np.asarray(getattr(m,k)[ids]).copy()for k in('cam_pos','cam_quat','cam_fovy')};raw=scene.raw
        frame=self.render_native(scene)
        camera_equal=all(np.array_equal(v,np.asarray(getattr(m,k)[ids]))for k,v in before.items())
        raw_untouched=scene.raw is raw
        if not camera_equal or not raw_untouched:raise RuntimeError('offline HD changed public camera model or original observation object')
        if not hasattr(self,'public_model_frame_checks'):self.public_model_frame_checks=[]
        self.public_model_frame_checks.append(dict(capture=self.captures,public_camera_model_equal=camera_equal,original_observation_object_untouched=raw_untouched,forced_observation_refresh=False))
        return frame
class LateralRecorder(PublicModelGuardRecorder):
    def render_native(self,scene):return lateral.LateralPrefixRecorder.native_frame(self,scene)
    def begin_segment(self,scene,**kwargs):
        with patch.object(original.observer,'POSE',lateral.POSE):return ORIGINAL_RECORDER.begin_segment(self,scene,**kwargs)
class IsolatedR104Recorder(PublicModelGuardRecorder):
    def begin_segment(self,scene,**kwargs):
        with patch.object(original.measured,'RGB_CHANNEL_CAP',64):result=super().begin_segment(scene,**kwargs)
        self.segment[-1]['isolated_RGB_bound_evidence']=dict(path=str(DIAGNOSTIC),sha256=hashlib.sha256(DIAGNOSTIC.read_bytes()).hexdigest(),measured_max_channels=42,cap_channels=64,cap_abs_delta=1,common_guard_remains20=True,online_protocol_unchanged=True)
        return result
def sidecars(instance):
    capture,final=paths(instance)
    for p in [capture/'report.json',final.with_suffix('.json')]:
        if not p.exists():continue
        d=json.loads(p.read_text())
        if d.get('simulator_pid')!=os.getpid():continue
        d.update(source_code_version='frozen-R1-T1+thin-display-only-HD-v5',display_pose=lateral.POSE if instance!='R1-04'else original.observer.POSE,
            initial_display_rgb_cap_channels=64 if instance=='R1-04' else 20,initial_display_rgb_max_delta=1,
            common_production_guard_unchanged20=True,online_protocol_changed=False,
            limitation='No additional per-stage forced RGB-D refresh. Initial same-state guard and per-render integration/public camera model/raw-object checks do not establish whole-trajectory RGB-bit identity.',
            blocked_stage_refresh_candidate=str(ROOT/'research/v4/candidates/scene_review_r1_full_hd_v4_20260926/HANDOFF.md'))
        p.write_text(json.dumps(d,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--instance',choices=list(original.CONFIG),required=True)
    ap.add_argument('--egl-device',type=int,default=0);ap.add_argument('--physical-gpu',type=int,default=0);ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args()
    recorder_cls=IsolatedR104Recorder if a.instance=='R1-04'else LateralRecorder
    try:
        with patch.object(original,'hashes',hashes),patch.object(original,'paths',paths),patch.object(original.measured,'MeasuredFreeRecorder',recorder_cls):original.main()
    finally:
        if not a.preflight_only:sidecars(a.instance)
if __name__=='__main__':main()
