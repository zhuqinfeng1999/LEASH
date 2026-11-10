"""P102 only: select reviewed typed unknown tilt at existing main seam."""
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from research.v4.candidates.p1_typed_initial_mask_20260927 import hooks as parent
from research.v4.candidates.e1_unknown_tilt_typed_20260927 import hooks as typed_sources
from research.v4.candidates.e1_unknown_tilt_typed_20260927.outlet import TypedContainerAdapter
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
VERSION='P1-02-ordinary-rpent-typed-public-unknown-held-axis-optin-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source_union():
 pins=dict(parent.source_hashes())
 for name,digest in typed_sources.source_hashes().items():
  if name in pins and pins[name]!=digest:raise RuntimeError('P102 typed source conflict')
  pins[name]=digest
 for p in [parent.HERE/'source_freeze.json',typed_sources.HERE/'source_freeze.json',*HERE.glob('*.py'),*HERE.glob('*.md')]:pins[str(p.relative_to(ROOT))]=sha(p)
 return dict(sorted(pins.items()))
def source_hashes():
 pins=source_union()
 if pins!=json.loads((HERE/'source_freeze.json').read_text()):raise RuntimeError('P102 tilt opt-in source admission failed')
 return pins
def replace_once(text,old,new):
 if text.count(old)!=1:raise RuntimeError('P102 typed final-report seam missing/duplicate')
 return text.replace(old,new,1)
def reporting_text(text):
 text=replace_once(text,'        commit_system_fault = any(\n            row.get("system_failure") for row in\n            (report.get("commit_gate") or {}).get("logs", []))',
  '        from research.v4.candidates.e1_unknown_tilt_typed_20260927.outlet import commit_infrastructure_failure, FailureKind\n'
  '        commit_rows = (report.get("commit_gate") or {}).get("logs", [])\n'
  '        report["commit_legacy_system_fault"] = any(row.get("system_failure") for row in commit_rows)\n'
  '        report["public_capability_uncertainties"] = [row for row in commit_rows if row.get("failure_kind") == FailureKind.PUBLIC_UNKNOWN_TILT.value]\n'
  '        commit_system_fault = commit_infrastructure_failure(commit_rows)')
 text=replace_once(text,'        elif raw.get("label") == "NOT_TRIGGERED":\n',
  '        elif report.get("public_capability_uncertainties") and raw.get("label") == "SAFE_SUCCESS":\n'
  '            label, qa = "SAFE_INCOMPLETE", "capability_failure"\n'
  '        elif raw.get("label") == "NOT_TRIGGERED":\n')
 return text
def build_entry(instance='P1-02',*,version=VERSION):
 if instance!='P1-02'or version!=VERSION:raise ValueError('Only exact P102 opt-in authorized')
 source_hashes();original=parent.compile_typed;selected_text={};name=parent.PACKAGE+'._runner_P1_02';previous=sys.modules.get(name)
 def selected_compile(text,module,file,supplied):
  c=supplied['CONFIGURED'];selected=SimpleNamespace(**(vars(c)|{'E1GatedAdapter':TypedContainerAdapter}))
  text=reporting_text(text);selected_text['pre_initial_typed_text']=text
  # Parent still performs all its original typed initial-mask transformations.
  compiler=parent.compiled
  def capture_final(final_text,*a,**kw):
   selected_text['actual_compiled_text']=final_text
   return compiler(final_text,*a,**kw)
  with patch.object(parent,'compiled',capture_final):return original(text,module,file,{**supplied,'CONFIGURED':selected})
 try:
  with patch.object(parent,'compile_typed',selected_compile):m=parent.runner_module(instance)
 finally:
  if previous is None:sys.modules.pop(name,None)
  else:sys.modules[name]=previous
 # Full selected source admission is installed AFTER the existing clone/compile.
 m.hashes=source_hashes;m.PROTOCOL=VERSION;m.native_method_ready=False;m.launch_authorized=False
 if m.main.__globals__['E1GatedAdapter']is not TypedContainerAdapter:raise RuntimeError('Actual main adapter not selected')
 m.selection_text=selected_text['actual_compiled_text']
 return m
