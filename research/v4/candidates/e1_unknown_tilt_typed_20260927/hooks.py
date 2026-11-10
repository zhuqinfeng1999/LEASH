"""Isolated generated-main classification only; parent421 remains immutable."""
import ast,hashlib,json
from pathlib import Path
from research.v4.candidates.e1_container_unknown_tilt_dispatch_20260927 import hooks as parent
from research.v4.candidates.e1_plain_config_dispatch_20260927.factory import compiled as original_compile
from research.v4.candidates.e1_02_plain_rpent_entry_20260927.hooks import replace_once
from .factory import components
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PACKAGE='research.v4.candidates.e1_unknown_tilt_typed_20260927'
VERSION='e1-own-container-typed-public-unknown-tilt-v42-candidate-v3'
def source_hashes():
    pins=dict(parent.source_hashes())
    paths=[parent.HERE/'source_freeze.json',*HERE.glob('*.py'),*HERE.glob('*.md'),
           *[p for p in HERE.glob('*.json')if p.name!='source_freeze.json']]
    for p in paths:pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    pins=dict(sorted(pins.items()));f=HERE/'source_freeze.json'
    if f.exists() and json.loads(f.read_text())!=pins:raise RuntimeError('Typed outlet source drift')
    return pins
def compiled(source,name,file,*args,**kwargs):
    source=replace_once(source,'        commit_system_fault = any(\n'
        '            row.get("system_failure") for row in\n'
        '            (report.get("commit_gate") or {}).get("logs", []))',
        '        from '+PACKAGE+'.outlet import commit_infrastructure_failure, FailureKind\n'
        '        commit_rows = (report.get("commit_gate") or {}).get("logs", [])\n'
        '        report["commit_legacy_system_fault"] = any(row.get("system_failure") for row in commit_rows)\n'
        '        report["public_capability_uncertainties"] = [row for row in commit_rows\n'
        '            if row.get("failure_kind") == FailureKind.PUBLIC_UNKNOWN_TILT.value]\n'
        '        commit_system_fault = commit_infrastructure_failure(commit_rows)')
    return original_compile(source,name,file,*args,**kwargs)
def runner_module(instance):
    # Reuse parent isolated AST recipe with only final compile callback changed.
    outer=dict(vars(parent),HERE=HERE,PACKAGE=PACKAGE,VERSION=VERSION,
        components=components,source_hashes=source_hashes)
    text=Path(parent.original.__file__).read_text();tree=ast.parse(text)
    outer.update(vars(parent.original));outer.update(HERE=HERE,PACKAGE=PACKAGE,VERSION=VERSION,
        components=components,source_hashes=source_hashes,compiled=compiled)
    for name in ('configure','runner_module'):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
        exec(compile(ast.get_source_segment(text,node),str(parent.original.__file__)+' [typed v3]', 'exec'),outer)
    return outer['runner_module'](instance)
