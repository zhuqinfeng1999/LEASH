"""Compile existing S1 boundaries with semantic labels, no new executor."""
from research.v4.leash_v43.run_manifest import source_custody_enabled
import ast,copy,hashlib,re,sys
from functools import lru_cache
from types import ModuleType,SimpleNamespace
from .config import binding,ROOT
from research.v4.candidates.e1_02_plain_rpent_entry_20260927.factory import PlainRPentToolkit
PREFIX='research.v4.candidates.s1_own_method_dispatch_20260927'
BASE='research/v4/candidates/s1_multiround_driver_20260925/'
def labels(value,b):
    return re.sub(r'\b(?:frying pan|hot pan|pan)\b',lambda m:b.noun,value)
def algorithm_sha(tree):
    class Erase(ast.NodeTransformer):
        def visit_Constant(self,n):
            if isinstance(n.value,str):n.value='TEXT'
            return n
        def visit_ImportFrom(self,n):return None
    return hashlib.sha256(ast.dump(Erase().visit(copy.deepcopy(tree)),include_attributes=False).encode()).hexdigest()
def compile_module(path,b,*,suffix,supplied=None):
    original=ast.parse((ROOT/path).read_text());tree=copy.deepcopy(original) if source_custody_enabled() else original;changes=[];supplied=supplied or{}
    class Bind(ast.NodeTransformer):
        def visit_ImportFrom(self,n):
            n.names=[x for x in n.names if(x.asname or x.name)not in supplied]
            return n if n.names else None
        def visit_Constant(self,n):
            if isinstance(n.value,str):
                value=labels(n.value,b)
                if n.value!=value:changes.append((n.value,value));n.value=value
            return n
    tree=Bind().visit(tree);ast.fix_missing_locations(tree)
    name=PREFIX+'._'+b.instance.replace('-','_')+'_'+suffix
    m=ModuleType(name);m.__file__=str(ROOT/path);m.__package__='research.v4.candidates.s1_multiround_driver_20260925'
    sys.modules[name]=m;m.__dict__.update(supplied)
    exec(compile(tree,str(ROOT/path)+' [own S1 semantic binding]','exec'),m.__dict__)
    if source_custody_enabled():
        m.semantic_changes=changes;m.algorithm_sha256=algorithm_sha(tree)
        if m.algorithm_sha256!=algorithm_sha(original):raise ValueError('nonsemantic algorithm changed')
        m.original_source_sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    return m
@lru_cache(maxsize=5)
def components(instance):
    b,instructions,Scene=binding(instance)
    bridge=compile_module(BASE+'bridge.py',b,suffix='bridge',supplied={'INSTRUCTIONS':instructions})
    park=compile_module(BASE+'park.py',b,suffix='park',supplied={'RecoveryRecord':bridge.RecoveryRecord})
    recovery=compile_module(BASE+'recovery.py',b,suffix='recovery',supplied={'RecoveryRecord':bridge.RecoveryRecord,'park_to_public_start':park.park_to_public_start})
    grounding=compile_module(BASE+'public_grounding.py',b,suffix='grounding')
    action=compile_module(BASE+'action_boundary.py',b,suffix='action')
    wiring=compile_module(BASE+'native_wiring.py',b,suffix='wiring',supplied={'HistoryEntry':bridge.HistoryEntry,'TaskGateDecision':bridge.TaskGateDecision})
    core=compile_module(BASE+'method_runner.py',b,suffix='core',supplied=dict(
        INSTRUCTIONS=instructions,S1ActionBoundary=action.S1ActionBoundary,
        RecoveryRecord=bridge.RecoveryRecord,run_s1_b0_candidate=bridge.run_s1_b0_candidate,
        recover_between_s1_rounds=recovery.recover_between_s1_rounds,
        S1MotionReviewer=grounding.S1MotionReviewer,S1PublicWorkspace=grounding.S1PublicWorkspace,
        normalize_s1_task_decision=grounding.normalize_s1_task_decision,
        S1OwnerCallbacks=wiring.S1OwnerCallbacks,S1TaskGateAdapter=wiring.S1TaskGateAdapter,
        callback_advance=wiring.callback_advance,callback_recovery=wiring.callback_recovery,
        sync_public_rpent_round_observation=wiring.sync_public_rpent_round_observation))
    dual=compile_module('research/v4/candidates/s1_native_cli_v5_typed_place_20260925/dual_view_workspace.py',b,suffix='dual',
        supplied={'PROMPTS':grounding.PROMPTS,'S1PublicWorkspace':grounding.S1PublicWorkspace})
    worker=compile_module('research/v4/candidates/s1_native_cli_v8_grasp_semantics_20260925/s1_review_worker.py',b,suffix='worker')
    from research.v4.candidates.s1_v42_tabletop_20260925.evaluator import S1Evaluator
    class Evaluator(S1Evaluator):
        def finish(self,*a,**kw):return {**super().finish(*a,**kw),'task_id':instance}
    return SimpleNamespace(binding=b,instructions=instructions,Scene=Scene,Evaluator=Evaluator,
        bridge=bridge,park=park,recovery=recovery,grounding=grounding,action=action,wiring=wiring,
        core=core,dual=dual,worker=worker,PlainRPentToolkit=PlainRPentToolkit,
        native_binding_ready=False,natural_method_ready=False)
