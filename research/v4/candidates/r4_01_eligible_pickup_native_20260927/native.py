"""Existing native stage driver, only own R401 script/config/source binding."""
import inspect,json,hashlib
from pathlib import Path
from types import SimpleNamespace
from research.v4.candidates.r4_derived_first_risk_native_20260926 import native as parent
from research.v4.candidates.r4_01_eligible_pickup_stop_20260927 import scripts,qualify
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
class Configuration:
 def __init__(self,instance):
  if instance!='R4-01':raise ValueError('exact R401 only')
  self.Scene=scripts.parent.Scene;self.namespace=scripts.namespace;self.scene_token='Scene';self.run_case=scripts.run_case
engine=SimpleNamespace(Configuration=Configuration,HERE=qualify.HERE)
def check_qualification():
 q=json.loads((qualify.HERE/'qualification30_summary.json').read_text())
 if q['source_sha256']!=qualify.hashes()or q['script_cases_passed']!=30 or not q['unchanged_initial_geometry_only_reused']['passed']:raise RuntimeError('own new script qualification/source required')
 return q
def hashes():
 pins={**parent.hashes(),**qualify.hashes()}
 for p in [*HERE.glob('*.py'),HERE/'protocol.json',qualify.HERE/'source_freeze.json',qualify.HERE/'qualification30_summary.json',qualify.HERE/'first3.json',qualify.HERE/'followup7.json']:
  pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 frozen=HERE/'source_freeze.json'
 if frozen.exists()and pins!=json.loads(frozen.read_text()):raise RuntimeError('own native source drift')
 return pins
source=inspect.getsource(parent.main)
old=""" qualification=json.loads((engine.HERE/a.instance/'qualification30_summary.json').read_text())
 if qualification['source_sha256']!=preflight.hashes() or qualification['script_cases_passed']!=30 or qualification['initial_geometry_passed']!=10:raise RuntimeError('Exact current source qualification required')"""
if source.count(old)!=1:raise RuntimeError('native qualification binding source drift')
source=source.replace(old,' qualification=check_qualification()').replace("choices=['R4-02','R4-05']","choices=['R4-01']").replace('controls_cap=1200','controls_cap=650')
namespace={**vars(parent),'HERE':HERE,'ROOT':ROOT,'engine':engine,'hashes':hashes,'check_qualification':check_qualification}
exec(compile(source,__file__,'exec'),namespace);main=namespace['main']
if __name__=='__main__':raise SystemExit(main())
