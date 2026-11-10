"""Single predeclared lateral offline camera, same frozen physical prefixes."""
import hashlib,json,math
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.candidates.scene_review_r1_public_observer_v2_20260926 import preview as prior
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PRIOR_HERE=prior.HERE;PRIOR_HASHES=prior.hashes
POS=np.array([.85,.60,1.47]);LOOK=np.array([0.,0.,1.08]);DELTA=POS-LOOK
POSE=dict(renderer_camera='MjvCamera free offline lateral observer',position_world_m=POS.tolist(),
          look_at_world_m=LOOK.tolist(),distance_m=float(np.linalg.norm(DELTA)),
          azimuth_deg=math.degrees(math.atan2(DELTA[1],DELTA[0])),
          elevation_deg=-math.degrees(math.asin(DELTA[2]/np.linalg.norm(DELTA))),
          fovy_deg=55.,display_version='opposite-robot-positiveX-positiveY-fixed-lateral-v3',online=False)
def hashes(instance):
    with patch.object(prior,'HERE',PRIOR_HERE):pins=PRIOR_HASHES(instance)
    for p in [*HERE.glob('*.py'),HERE/'protocol.json']:
        pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    pins=dict(sorted(pins.items()));f=HERE/'source_freeze.json'
    if f.exists()and pins!=json.loads(f.read_text())[instance]:raise ValueError('single lateral camera candidate source drift')
    return pins
class LateralPrefixRecorder(prior.PrefixPreviewRecorder):
    def native_frame(self,scene):
        o=prior.original.observer
        with patch.object(o,'POS',POS),patch.object(o,'LOOK',LOOK),patch.object(o,'POSE',POSE):
            return o.FreeObserverRecorder.native_frame(self,scene)
    def begin_segment(self,scene,**kwargs):
        with patch.object(prior.original.observer,'POSE',POSE):
            return prior.original.measured.MeasuredFreeRecorder.begin_segment(self,scene,**kwargs)
def main():
    with patch.object(prior,'HERE',HERE),patch.object(prior,'hashes',hashes),patch.object(prior,'PrefixPreviewRecorder',LateralPrefixRecorder):
        return prior.main()
if __name__=='__main__':raise SystemExit(main())
