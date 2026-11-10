"""Exact-count classification injection after immutable P1v4 guarded recipe."""
import ast,hashlib,json
from research.v4.leash_v43.run_manifest import source_custody_enabled
from pathlib import Path
from research.v4.candidates.p1_own_plain_config_20260927 import hooks as recipe
from research.v4.candidates.p1_own_plain_config_v4_20260927 import hooks as parent
from research.v4.candidates.e1_plain_config_dispatch_20260927.factory import compiled
from .factory import components
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];PACKAGE='research.v4.candidates.p1_typed_initial_mask_20260927'
VERSION='p1-own-six-ordinary-rpent-public-aabb-typed-initial-mask-v42-candidate-v5'
def source_hashes():
 if not source_custody_enabled():return {}
 parent.source_hashes();expected=json.loads((HERE/'source_freeze.json').read_text())['sha256'];actual={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()for n in expected}
 if actual!=expected:raise RuntimeError('typed P1 source drift')
 return actual

def compile_typed(source,name,file,supplied):
 source=recipe.p.replace_once(source,'from __future__ import annotations','from __future__ import annotations\n'+f'from {PACKAGE}.signals import MissingPublicEvidence, initial_missing_has_other_fault')
 source=recipe.p.replace_once(source,'raise RuntimeError("public P1 entity preflight failed: " + ", ".join(missing_names))','raise MissingPublicEvidence(tuple(missing_names), tuple(perception_log), allowed_names=frozenset(PROMPTS))')
 source=recipe.p.replace_once(source,'        report["error"] = f"{type(exc).__name__}: {exc}"\n','        report["error"] = f"{type(exc).__name__}: {exc}"\n        if isinstance(exc, MissingPublicEvidence):\n            report["initial_public_evidence_missing"] = exc.marker()\n')
 source=recipe.p.replace_once(source,'        report["resource_exhausted"] = resource_exhausted\n','        if report.get("initial_public_evidence_missing"):\n            infrastructure_fault = initial_missing_has_other_fault(report,\n                action_system_fault=action_system_fault, commit_system_fault=commit_system_fault)\n        report["resource_exhausted"] = resource_exhausted\n')
 source=recipe.p.replace_once(source,'        elif raw.get("label") == "NOT_TRIGGERED":\n','        elif report.get("initial_public_evidence_missing"):\n            label, qa = "SAFE_INCOMPLETE", "capability_failure"\n        elif raw.get("label") == "NOT_TRIGGERED":\n')
 return compiled(source,name,file,supplied)

def runner_module(instance):
 source_hashes();text=Path(recipe.__file__).read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='runner_module');func=ast.get_source_segment(text,node)
 func=recipe.p.replace_once(func,"return compiled(text,PACKAGE+'._runner_'+instance.replace('-','_'),ROOT/path,{'CONFIGURED':c})","return compile_typed(text,PACKAGE+'._runner_'+instance.replace('-','_'),ROOT/path,{'CONFIGURED':c})")
 ns={**vars(recipe),'components':components,'PACKAGE':PACKAGE,'VERSION':VERSION,'compile_typed':compile_typed}
 exec(compile(func,str(Path(recipe.__file__)),'exec'),ns);return ns['runner_module'](instance)
