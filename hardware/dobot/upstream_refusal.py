"""Instance-local port of upstream's documented Decisions refusal repair.

Exact upstream method bodies are loaded from hash-verified source snapshots.
The handoff's frozen modules, request/prompt construction and shared globals
stay unchanged. A refusal never supplies a probability or authorizes a send.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import FunctionType, MethodType

from research.v4.development.confidence_gate_dev_20261006 import core as original_core
from research.v4.development.confidence_gate_dev_20261006 import runtime as original_runtime
from research.v4.development.decision_gate_dev_20261019 import backend as original_backend

SOURCE_ROOT=Path(__file__).with_name('upstream_refusal_repair3')
SOURCE_COMMIT='145678e70752d27ff919326e14b74aaf7ad03cc1'
_manifest=json.loads((SOURCE_ROOT/'manifest.json').read_text(encoding='utf-8'))
if _manifest['commit']!=SOURCE_COMMIT:
    raise ValueError('Unrecognized upstream refusal repair commit')


def provenance():
    return dict(commit=SOURCE_COMMIT,backend_version=BACKEND_VERSION,
        manifest_sha256=hashlib.sha256((SOURCE_ROOT/'manifest.json').read_bytes()).hexdigest(),
        source_sha256={name:row['sha256'] for name,row in _manifest['files'].items()})


def _source(name):
    path=SOURCE_ROOT/name
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=_manifest['files'][name]['sha256']:
        raise ValueError('Upstream refusal repair source hash mismatch: '+name)
    return path,ast.parse(raw.decode('utf-8'))


def _node(tree,name,owner=None):
    body=tree.body
    if owner is not None:
        body=next(node for node in body if isinstance(node,ast.ClassDef) and node.name==owner).body
    return next(node for node in body if isinstance(node,ast.FunctionDef) and node.name==name)


def _method(file,name,reference,owner=None,overrides=None):
    path,tree=_source(file)
    node=_node(tree,name,owner)
    if node.decorator_list:
        raise ValueError('Unexpected decorated upstream method')
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    namespace=dict(reference.__globals__,**(overrides or {}))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,node],type_ignores=[])),str(path),'exec'),namespace)
    return namespace[name]


_backend_path,_backend_tree=_source('backend.py')
BACKEND_VERSION=next(ast.literal_eval(node.value) for node in _backend_tree.body
    if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='VERSION' for target in node.targets))
normalize=_method('backend.py','normalize',original_backend.normalize,overrides={'VERSION':BACKEND_VERSION})

# Successful requests and the scoring transport body are exactly the frozen
# implementation. Only this instance's normalizer and version are replaced.
_old_score=original_backend.DecisionBackend.score
_old_tree=ast.parse(Path(inspect.getsourcefile(_old_score)).read_text(encoding='utf-8'))
if ast.dump(_node(_old_tree,'score','VisualProbabilityBackend'),include_attributes=False)!=ast.dump(
        _node(_backend_tree,'score','VisualProbabilityBackend'),include_attributes=False):
    raise ValueError('Upstream scoring transport changed; separate port review required')


def install_backend(backend):
    if type(backend) not in {original_backend.DecisionBackend,original_backend.JevBackend}:
        raise ValueError('Explicit frozen visual probability backend instance required')
    function=FunctionType(_old_score.__code__,dict(_old_score.__globals__,normalize=normalize,
        VERSION=BACKEND_VERSION),_old_score.__name__,_old_score.__defaults__,_old_score.__closure__)
    function.__kwdefaults__=_old_score.__kwdefaults__
    backend.score=MethodType(function,backend)
    backend.hardware_refusal_repair=dict(provenance(),
        scope='Typed Decisions refusal classification; no probability, policy or threshold substitution')
    return backend


class GateCore(original_core.GateCore):
    score=_method('core.py','score',original_core.GateCore.score,'GateCore')
    feedback=_method('core.py','feedback',original_core.GateCore.feedback,'GateCore')


review=_method('runtime.py','_review',original_runtime.ConfidenceAdapter._review,'ConfidenceAdapter')
