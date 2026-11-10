"""Opt-in bindings around existing code objects; no copied runner/executor."""
from research.v4.leash_v43.run_manifest import source_custody_enabled
from functools import lru_cache
from pathlib import Path
from types import FunctionType,ModuleType
import hashlib,json
from research.v4.candidates.s1_s2_sdk_history_boundary_20260927.deferred_context import DeferredPlannerSession,DeferredBoundary
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
VERSION='deferred-public-round-context-optin-v1'

def clone_function(function,globals_):
    value=FunctionType(function.__code__,globals_,function.__name__,function.__defaults__,function.__closure__)
    value.__kwdefaults__=function.__kwdefaults__;value.__annotations__=function.__annotations__
    value.__qualname__=function.__qualname__
    return value

def clone_module(original,**bindings):
    value=ModuleType(original.__name__+'_deferred_optin');value.__dict__.update(vars(original));value.__dict__.update(bindings)
    # Rebind only functions defined by this module to the same code object and
    # new globals. Imported helpers remain their original implementation.
    for name,function in list(vars(original).items()):
        if isinstance(function,FunctionType)and function.__globals__ is vars(original):
            value.__dict__[name]=clone_function(function,value.__dict__)
    return value

def session_constructor(original):
    def construct(*args,**kwargs):return DeferredPlannerSession(original(*args,**kwargs))
    construct.original_constructor=original
    return construct

def boundary_constructor(original):
    def construct(inner,*args,**kwargs):
        if not isinstance(inner,DeferredPlannerSession):raise TypeError('explicit deferred production session required')
        return DeferredBoundary(original(inner,*args,**kwargs),inner)
    construct.original_constructor=original
    return construct

def source_hashes():
    if not source_custody_enabled():
        return {}
    parent=ROOT/'research/v4/candidates/s1_s2_sdk_history_boundary_20260927/source_freeze.json'
    hashes=json.loads(parent.read_text())
    for name,sha in hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('qualified history source drift: '+name)
    hashes[str(parent.relative_to(ROOT))]=hashlib.sha256(parent.read_bytes()).hexdigest()
    for p in HERE.glob('*.py'):hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    frozen=HERE/'source_freeze.json'
    if frozen.exists()and json.loads(frozen.read_text())!=dict(sorted(hashes.items())):raise RuntimeError('opt-in production source drift')
    return dict(sorted(hashes.items()))

@lru_cache(maxsize=5)
def s1_components(instance):
    from research.v4.candidates.s1_own_method_dispatch_20260927.factory import components
    c=components(instance)
    bridge=clone_module(c.bridge,S1PersistentSession=boundary_constructor(c.bridge.S1PersistentSession))
    core=clone_module(c.core,run_s1_b0_candidate=bridge.run_s1_b0_candidate,
        CodexPersistentSession=session_constructor(c.core.CodexPersistentSession),source_hashes=source_hashes)
    return c,bridge,core

@lru_cache(maxsize=1)
def s2_core_chain():
    from research.v4.candidates.s3_v42_tabletop_20260926 import method_core as original_shared
    from research.v4.candidates.s2_higher_method_20260926 import method_core as original_old
    from research.v4.candidates.s2_future_production_dispatch_20260927 import factory as original_dispatch
    from research.v4.candidates.s2_future_method_v2_20260927 import core as original_core
    shared=clone_module(original_shared,S3PersistentWrapper=boundary_constructor(original_shared.S3PersistentWrapper))
    old=clone_module(original_old,shared=shared,
        JournaledS3CodexSession=session_constructor(original_old.JournaledS3CodexSession))
    dispatch=clone_module(original_dispatch,original_core=old)
    core=clone_module(original_core,original_run_core=dispatch.run_core,old=old)
    return core,dispatch,old,shared

def runner_module(instance,*,enable_deferred_history=False):
    """Explicit opt-in; False returns the unchanged current original runner."""
    if instance.startswith('S1-'):
        from research.v4.candidates.s1_own_method_dispatch_20260927 import hooks as original_hooks
        original=original_hooks.runner_module(instance)
        if not enable_deferred_history:return original
        _c,bridge,core=s1_components(instance)
        value=clone_module(original,run_s1_method_core=core.run_s1_method_core,source_hashes=source_hashes)
        def preflight(a,**kwargs):
            output,_=original.preflight(a,**kwargs)
            return output,source_hashes()
        value.preflight=preflight;value.optin_core=core;value.optin_bridge=bridge
    elif instance.startswith('S2-'):
        from research.v4.candidates.s2_future_method_v2_20260927 import run_method as original
        from research.v4.candidates.s2_future_production_dispatch_20260927.factory import binding
        binding(instance)
        if not enable_deferred_history:return original
        # Existing generated main imports .core/.selection/.review_worker.
        # Only that package resolution moves to this opt-in alias package.
        globals_={**original.main.__globals__,'__package__':__package__,'hashes':source_hashes}
        value=ModuleType(__package__+'._'+instance.replace('-','_'));value.__dict__.update(globals_)
        value.main=clone_function(original.main,value.__dict__)
        value.optin_core,value.optin_dispatch,value.optin_old,value.optin_shared=s2_core_chain()
        value.compiled_main_sha256=hashlib.sha256(original.source.encode()).hexdigest()
    else:raise ValueError('only own registered S1/S2 instances')
    value.PROTOCOL=original.PROTOCOL+'-'+VERSION if hasattr(original,'PROTOCOL')else original.VERSION+'-'+VERSION
    value.deferred_history_opt_in=True;value.natural_method_ready=False
    return value
