"""Native free observer camera; model/data/public physics are untouched."""
import hashlib,json,math,time
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import mujoco,numpy as np
from PIL import Image
from research.v4.candidates.scene_review_export_20260926 import export_p1_02 as base,recorder as recording
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];OLD=ROOT/'research/v4/candidates/scene_review_export_20260926'
POS=np.array([.10,-.28,2.15]);LOOK=np.array([.10,.04,.95]);DELTA=POS-LOOK
POSE=dict(renderer_camera='MjvCamera free offline observer',position_world_m=POS.tolist(),look_at_world_m=LOOK.tolist(),distance_m=float(np.linalg.norm(DELTA)),azimuth_deg=math.degrees(math.atan2(DELTA[1],DELTA[0])),elevation_deg=-math.degrees(math.asin(DELTA[2]/np.linalg.norm(DELTA))),fovy_deg=55.,display_version='high-oblique-free-render-camera-v3',online=False)
BINDING=ROOT/'envs/libero/lib/python3.11/site-packages/robosuite/utils/binding_utils.py'
def hashes():
    old=json.loads((OLD/'source_freeze.json').read_text());pins={}
    for name,sha in old.items():
        path=Path(name)if name.startswith('/')else ROOT/name;actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=sha:raise ValueError('original94-source drift: '+name)
        pins[name]=actual
    for p in[*HERE.glob('*.py'),HERE/'protocol.json',OLD/'source_freeze.json',BINDING]:pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    frozen=HERE/'source_freeze.json'
    if frozen.exists()and pins!=json.loads(frozen.read_text()):raise ValueError('free observer source drift')
    return pins
class FreeObserverRecorder(recording.NativeSceneRecorder):
    def native_frame(self,scene):
        state=self._state(scene);sim=scene.backend.sim;context=sim._render_context_offscreen
        if context is None:raise RuntimeError('existing native render context missing')
        attrs=('type','fixedcamid','trackbodyid','lookat','distance','azimuth','elevation');old={k:np.array(getattr(context.cam,k),copy=True)if k=='lookat'else getattr(context.cam,k)for k in attrs};globalfovy=float(sim.model.vis.global_.fovy)
        def render():
            context.cam.type=mujoco.mjtCamera.mjCAMERA_FREE;context.cam.fixedcamid=-1;context.cam.trackbodyid=-1;context.cam.lookat[:]=LOOK;context.cam.distance=POSE['distance_m'];context.cam.azimuth=POSE['azimuth_deg'];context.cam.elevation=POSE['elevation_deg'];sim.model.vis.global_.fovy=POSE['fovy_deg']
            return sim.render(width=self.width,height=self.height,camera_name=None)[::-1]
        started=time.monotonic()
        try:
            try:frame=render()
            except BaseException as exc:
                self.resolution_attempts.append(dict(width=self.width,height=self.height,error=f'{type(exc).__name__}: {exc}'))
                if self.pipe is not None or(self.width,self.height)!=(1920,1080):raise
                self.width,self.height=1280,720;frame=render()
            actual_gl_pose=dict(position_world_m=((context.scn.camera[0].pos+context.scn.camera[1].pos)/2).tolist(),forward_world=((context.scn.camera[0].forward+context.scn.camera[1].forward)/2).tolist())
        finally:
            for k,v in old.items():
                if k=='lookat':context.cam.lookat[:]=v
                else:setattr(context.cam,k,v)
            sim.model.vis.global_.fovy=globalfovy
        self.render_seconds+=time.monotonic()-started;self.captures+=1
        if frame.shape!=(self.height,self.width,3)or frame.dtype!=np.uint8:raise ValueError('native observer shape; no upscaling')
        same=np.array_equal(state,self._state(scene))
        if not same:raise RuntimeError('free render changed integration state')
        if self.captures<=2 or self.captures%100==0:self.render_state_checks.append(dict(capture=self.captures,integration_state_equal=same,actual_gl_pose=actual_gl_pose))
        return np.ascontiguousarray(frame)
    def begin_segment(self,scene,**kwargs):
        sim=scene.backend.sim;publicids=[sim.model.camera_name2id(n)for n in scene.public_views];model=sim.model._model;modelbefore={k:np.array(getattr(model,k)[publicids],copy=True)for k in('cam_pos','cam_quat','cam_fovy')};before=scene.capture_agent();state=self._state(scene)
        super().begin_segment(scene,**kwargs)
        scene.raw=scene.backend._get_observations(force_update=True);after=scene.capture_agent();checks={n:{k:bool(np.array_equal(before['cameras'][n][k],after['cameras'][n][k]))for k in('rgb','depth_m','intrinsics','extrinsics')}for n in scene.public_views};model_same=all(np.array_equal(getattr(model,k)[publicids],v)for k,v in modelbefore.items());physics_same=np.array_equal(state,self._state(scene));proof=dict(integration_state_equal=bool(physics_same),public_camera_model_equal=model_same,public_buffers_and_calibration_equal=checks,pose=POSE)
        self.segment[-1]['offline_display_camera_validation']=proof
        (HERE/'capture'/f'chapter{len(self.segment)}_camera_guard.json').write_text(json.dumps(proof,indent=2)+'\n')
        if not physics_same or not model_same or not all(all(v.values())for v in checks.values()):raise RuntimeError('free observer changed physics/public images')
        Image.fromarray(self.native_frame(scene)).save(HERE/'capture'/f'chapter{len(self.segment)}_initial_native_spectator.png')
def main():
    with ExitStack()as scope:
        scope.enter_context(patch.object(base,'HERE',HERE));scope.enter_context(patch.object(base,'hashes',hashes));scope.enter_context(patch.object(recording,'NativeSceneRecorder',FreeObserverRecorder));base.main()
    path=base.FINAL.with_suffix('.json')
    if path.exists():
        d=json.loads(path.read_text());d.update(display_version=POSE['display_version'],offline_spectator_camera=POSE,prior_display_failure_evidence=[str(OLD/'capture/first_observer_angle_failure.json'),str(ROOT/'research/v4/candidates/scene_review_high_observer_20260926/capture/report.json'),str(ROOT/'research/v4/candidates/scene_review_observer_diagnosis_20260926/report.json')]);path.write_text(json.dumps(d,indent=2)+'\n');(HERE/'capture/report.json').write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':main()
