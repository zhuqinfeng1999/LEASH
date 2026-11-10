"""Opt-in five-own REAL S1 binding only; no CLI or launch authorization."""
from pathlib import Path
from functools import lru_cache
import hashlib,json,inspect
from research.v4.candidates.s1_s2_deferred_production_optin_20260927 import hooks
from research.v4.candidates.s1_own_method_dispatch_20260927 import hooks as own_hooks
from research.v4.candidates.s1_own_method_dispatch_20260927.config import IDS

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
APPROVED=ROOT/'research/v4/candidates/s1_deferred_native_integration_20260927/source_freeze.json'
VERSION='s1-five-own-real-deferred-optin-v42-source-preparation-v1'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load_expected():
    path=HERE/'source_freeze.json'
    return json.loads(path.read_text())if path.exists()else None
def admit(actual,expected):
    if expected is None or actual!=expected:raise RuntimeError('Real S1 opt-in source drift or missing freeze')
    return actual
def source_union():
    value=json.loads(APPROVED.read_text())
    for name,digest in value.items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Approved real S1 dependency drift: '+name)
    value[str(APPROVED.relative_to(ROOT))]=sha(APPROVED)
    for path in[*HERE.glob('*.py'),*HERE.glob('*.md')]:value[str(path.relative_to(ROOT))]=sha(path)
    return dict(sorted(value.items()))
def source_hashes():return admit(source_union(),load_expected())

@lru_cache(maxsize=5)
def components(instance):
    if instance not in IDS:raise ValueError('Unregistered real S1 instance')
    c,bridge,original_core=hooks.s1_components(instance)
    # New own globals, original code/bridge/session constructors. Explicit whole
    # admission assignment comes AFTER cloning; originals are never mutated.
    core=hooks.clone_module(original_core)
    core.source_hashes=source_hashes
    return c,bridge,core

def runner_module(instance):
    if instance not in IDS:raise ValueError('Unregistered real S1 instance')
    c,bridge,core=components(instance)
    template=hooks.runner_module(instance,enable_deferred_history=True)
    original=inspect.getclosurevars(template.preflight).nonlocals['original']
    value=hooks.clone_module(template,__package__=__package__,HERE=HERE)
    # Keep original preflight's output/service/engineering checks, replace only
    # its returned source packet. This closure does not run on module creation.
    def preflight(a,**kwargs):
        output,_=original.preflight(a,**kwargs)
        return output,source_hashes()
    value.preflight=preflight
    value.source_hashes=source_hashes
    value.run_s1_method_core=core.run_s1_method_core
    value.optin_core=core;value.optin_bridge=bridge;value.own_components=c
    value.original_template=template
    value.PROTOCOL=template.PROTOCOL+'-'+VERSION
    value.native_method_ready=False;value.launch_authorized=False
    value.production_transport_kind='original REAL SDK/reviewer/SAM/VLA'
    return value
