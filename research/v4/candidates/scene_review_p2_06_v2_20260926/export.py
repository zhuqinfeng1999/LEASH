"""Existing native recorder with measured one-level refresh RGB jitter bounds."""
import json,hashlib
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
import numpy as np
from PIL import Image
from research.v4.candidates.scene_review_p2_06_20260926 import export as base
from research.v4.candidates.scene_review_free_observer_20260926 import export_p1_02 as free
from research.v4.candidates.scene_review_export_20260926 import recorder as recording
from research.v4.candidates.scene_review_r3_02_render_diag_20260926.diagnose import compare
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
DIAG=ROOT/'research/v4/candidates/scene_review_p2_06_noise_diag_20260926/forced_refresh_report.json'
RGB_CHANNEL_CAP=20;RGB_DELTA_CAP=1
ORIGINAL_BASE_HASHES=base.hashes
ORIGINAL_BASE_HERE=base.HERE
def hashes():
 with patch.object(base,'HERE',ORIGINAL_BASE_HERE):out=ORIGINAL_BASE_HASHES()
 for p in [*HERE.glob('*.py'),HERE/'protocol.json',DIAG,DIAG.parent/'forced_refresh.py',ROOT/'research/v4/candidates/scene_review_r3_02_render_diag_20260926/diagnose.py']:
  out[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 f=HERE/'source_freeze.json'
 if f.exists()and out!=json.loads(f.read_text()):raise ValueError('v2source drift')
 return out
class MeasuredFreeRecorder(free.FreeObserverRecorder):
 def begin_segment(self,scene,**kwargs):
  sim=scene.backend.sim;m=sim.model._model;ids=[sim.model.camera_name2id(n)for n in scene.public_views]
  state=self._state(scene);cam={k:np.asarray(getattr(m,k)[ids]).copy()for k in ['cam_pos','cam_quat','cam_fovy']};proprio={k:np.asarray(scene.raw[k]).copy()for k in ['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']}
  a=scene.capture_agent();scene.raw=scene.backend._get_observations(force_update=True);b=scene.capture_agent();baseline=compare(a,b)
  recording.NativeSceneRecorder.begin_segment(self,scene,**kwargs)
  scene.raw=scene.backend._get_observations(force_update=True);c=scene.capture_agent();actual=compare(b,c)
  def allowed(d):return all(v['rgb']['max_abs_delta']<=RGB_DELTA_CAP and v['rgb']['changed_elements']<=RGB_CHANNEL_CAP and all(v[k]['equal']for k in ['depth_m','intrinsics','extrinsics'])for v in d.values())
  exact_state=bool(np.array_equal(state,self._state(scene)));exact_cam=all(np.array_equal(v,np.asarray(getattr(m,k)[ids]))for k,v in cam.items());exact_proprio=all(np.array_equal(v,np.asarray(scene.raw[k]))for k,v in proprio.items())
  proof=dict(integration_state_equal=exact_state,public_camera_model_equal=exact_cam,proprioception_equal=exact_proprio,before_force_without_observer=baseline,before_after_observer_force=actual,display_only_rgb_noise_bound=dict(max_changed_channels=RGB_CHANNEL_CAP,max_abs_delta=RGB_DELTA_CAP,total_channels=480*640*3,measured_prior_max_changed_channels=10,diagnostic=str(DIAG.relative_to(ROOT))),comparison_passed=bool(exact_state and exact_cam and exact_proprio and allowed(baseline)and allowed(actual)),pose=free.POSE)
  self.segment[-1]['offline_display_camera_validation']=proof;(HERE/'capture'/f'chapter{len(self.segment)}_camera_guard.json').write_text(json.dumps(proof,indent=2)+'\n')
  if not proof['comparison_passed']:raise RuntimeError('observer modified physical/calibration/depth or exceeded measured RGB jitter')
  Image.fromarray(self.native_frame(scene)).save(HERE/'capture'/f'chapter{len(self.segment)}_initial_native_spectator.png')
def main():
 with ExitStack()as scope:
  scope.enter_context(patch.object(base,'HERE',HERE));scope.enter_context(patch.object(base,'hashes',hashes));scope.enter_context(patch.object(base.observer,'FreeObserverRecorder',MeasuredFreeRecorder));base.main()
if __name__=='__main__':main()
