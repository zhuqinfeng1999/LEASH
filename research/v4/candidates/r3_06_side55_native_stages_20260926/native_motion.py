"""Capture actual side-wrist completion and final approach, physics unchanged."""
import inspect,json,hashlib
from pathlib import Path
from research.v4.candidates.r3_06_side55_entry_20260926 import native_motion as parent
from research.v4.candidates.r3_06_side55_entry_20260926.qualify import hashes as qualification_hashes
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def hashes():
 pins=qualification_hashes()
 for p in HERE.glob('*.py'):pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 f=HERE/'source_freeze.json'
 if f.exists()and pins!=json.loads(f.read_text()):raise ValueError('native-stage-only source drift')
 return pins
SOURCE=parent.SOURCE
changes={
 "self._capture_closed=False;self._capture_lifted=False;self._capture_release=False;self._positive_held=0":"self._capture_closed=False;self._capture_lifted=False;self._capture_release=False;self._positive_held=0;self._was_held_rotation=False;self._capture_tilt=False;self._capture_preentry=False",
 "if float(action[6])<0 and self.grasp_assist.held and not self._capture_release:":"if self.grasp_assist.held and self._was_held_rotation and np.linalg.norm(action[3:6])==0 and not self._capture_tilt:\n                self._capture_tilt=True;snapshot(self,'after_actual_side_tilt')\n            if float(action[6])<0 and self.grasp_assist.held and not self._capture_release:",
 "result=super().step(action);control_count+=1":"result=super().step(action);control_count+=1;self._was_held_rotation=bool(self.grasp_assist.held and np.linalg.norm(action[3:6])>0)",
 "return result\n        def close(self):":"if self.grasp_assist.held and hasattr(self,'_native_preentry_goal') and np.linalg.norm(np.asarray(self.raw['robot0_eef_pos'])-self._native_preentry_goal)<.008 and not self._capture_preentry:\n                self._capture_preentry=True;snapshot(self,'actual_preentry')\n            return result\n        def close(self):",
 "with patch.dict(scripts.namespace,{'R3TabletopFacade':CaptureScene}):result=scripts.run_case(seed,'naive')":"original_prepare=scripts.namespace['_prepare_path']\n        def capture_prepare(scene,details):\n            points=original_prepare(scene,details);scene._native_preentry_goal=points[1].copy();return points\n        with patch.dict(scripts.namespace,{'R3TabletopFacade':CaptureScene,'_prepare_path':capture_prepare}):result=scripts.run_case(seed,'naive')",
 "['initial','actual_attachment','closed_actual','lifted_2cm','final_after_script']":"['initial','actual_attachment','closed_actual','lifted_2cm','after_actual_side_tilt','actual_preentry','final_after_script']"
}
for anchor,replacement in changes.items():
 if SOURCE.count(anchor)!=1:raise ValueError('finite stage capture anchor changed: '+anchor)
 SOURCE=SOURCE.replace(anchor,replacement)
namespace=dict(parent.namespace);namespace.update(HERE=HERE,ROOT=ROOT,hashes=hashes);exec(compile(SOURCE,__file__,'exec'),namespace);run_capture=namespace['run_capture'];exec(compile(inspect.getsource(parent.parent.main),__file__,'exec'),namespace);main=namespace['main']
if __name__=='__main__':raise SystemExit(main())
