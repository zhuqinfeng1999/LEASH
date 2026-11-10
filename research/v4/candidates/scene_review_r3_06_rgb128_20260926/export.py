"""Isolated R306 display guard based on independent same-state diagnostics."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch
from research.v4.candidates.scene_review_r3_06_side55_20260926 import export as old
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];FINAL=HERE/'R3-06.rgb128.preview.mp4'
DIAGNOSTIC=ROOT/'research/v4/candidates/r3_06_forced_refresh_repro_20260926/native_diagnostic/report.json'
GUARD=dict(max_changed_channels=128,max_abs_delta=1,total_channels=480*640*3,scope='Offline display consistency comparison only, not online observations',authorization='Root authorized isolated128-channel bound after actual no-observer wrist123-channel intrinsic jitter',exact_fields=['integration_state','public_camera_model','proprioception','depth_m','intrinsics','extrinsics'])
class DiagnosticBoundRecorder(old.Native1080Recorder):
 def begin_segment(self,scene,**kwargs):
  with patch.object(old.measured,'RGB_CHANNEL_CAP',128):result=super().begin_segment(scene,**kwargs)
  self.segment[-1]['independent_rgb_reproducibility_evidence']=dict(diagnostic_path=str(DIAGNOSTIC),diagnostic_sha256=hashlib.sha256(DIAGNOSTIC.read_bytes()).hexdigest(),bound=GUARD)
  return result
def hashes():
 pins=old.hashes()
 for p in[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'cpu_contract.json',DIAGNOSTIC,old.HERE/'capture/report.json']:
  pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 pins=dict(sorted(pins.items()));f=HERE/'source_freeze.json'
 if f.exists()and pins!=json.loads(f.read_text()):raise ValueError('R306 isolated128 source drift')
 return pins
def sidecars():
 for p in[HERE/'capture/report.json',FINAL.with_suffix('.json')]:
  if p.exists():
   d=json.loads(p.read_text());d.update(instance_id='R3-06',instruction=old.INSTRUCTION,source_code_version='r306-same-side55+isolated-independent-RGB128-display-guard',original_script_limits=dict(controls=1200,wall_s=90),control_cap_per_script=1200,no_full_tray_containment_door_closure_or_heating_claim=True,display_rgb_guard=GUARD,independent_diagnostic=dict(path=str(DIAGNOSTIC),physical_pid=4109727,wall_s=2.268,controls=0,source_pins=83),previous_failed_guard=str(old.HERE/'capture/report.json'),online_protocol_changed=False,guard_not_global=True);p.write_text(json.dumps(d,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--egl-device',type=int,default=0);p.add_argument('--physical-gpu',type=int,default=0);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();pins=hashes()
 if a.preflight_only:print(json.dumps(dict(source_count=len(pins),will_construct_scene=False,models_called=[],guard=GUARD,original_physics_unchanged=True)));return
 try:
  with ExitStack()as scope:
   for mod,k,v in[(old.measured,'HERE',HERE),(old.measured,'hashes',hashes),(old.measured,'MeasuredFreeRecorder',DiagnosticBoundRecorder),(old.driver,'scripts',old.ScriptAdapter),(old.driver,'qualify',SimpleNamespace(hashes=hashes)),(old.driver,'FINAL',FINAL)]:scope.enter_context(patch.object(mod,k,v))
   old.measured.main()
 finally:sidecars()
if __name__=='__main__':main()
