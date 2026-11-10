"""Thin own-instance binding of the existing v8 real main and two-round core."""
from research.v4.leash_v43.run_manifest import source_custody_enabled
import ast,hashlib,json,re,sys
from pathlib import Path
from types import ModuleType
from .config import ROOT,IDS,accepted_sources
from .factory import components,labels
from .reviewer import WORKERS
HERE=Path(__file__).resolve().parent
VERSION='s1-five-own-scenes-ordinary-rpent-persistent-two-round-v42-candidate-v1'
ORIGINAL='research/v4/candidates/s1_native_cli_v8_grasp_semantics_20260925/run_method.py'
def source_hashes():
    if not source_custody_enabled():
        return {}
    pins={}
    for instance in IDS:
        for name,sha in accepted_sources(instance).items():
            if name in pins and pins[name]!=sha:raise ValueError('own accepted union mismatch')
            pins[name]=sha
    from research.v4.candidates.s1_native_cli_v8_grasp_semantics_20260925 import run_method as original
    for name,sha in original.source_hashes().items():
        if name in pins and pins[name]!=sha:raise ValueError('legacy runtime union drift')
        pins[name]=sha
    extra=('research/v4/candidates/e1_02_plain_rpent_entry_20260927/factory.py',
           'research/v4/configs/instance_registry.json')
    for p in [*(ROOT/name for name in extra),*HERE.glob('*.py'),*HERE.glob('*.md'),
              *(p for p in HERE.glob('*.json')if p.name!='source_freeze.json')]:
        pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    pins=dict(sorted(pins.items()));freeze=HERE/'source_freeze.json'
    if freeze.exists()and json.loads(freeze.read_text())!=pins:raise ValueError('own S1 source drift')
    return pins
def runner_module(instance):
    c=components(instance);b=c.binding;path=ROOT/ORIGINAL;tree=ast.parse(path.read_text());ordinary=[]
    supplied={'run_s1_method_core':c.core.run_s1_method_core,'S1SDKReviewerWorker':WORKERS[instance],
        'S1DualViewWorkspace':c.dual.S1DualViewWorkspace,'S1StablePan300Facade':c.Scene,
        'S1Evaluator':c.Evaluator,'PanPickPlaceToolkit':c.PlainRPentToolkit,
        'install_public_name_aliases':lambda:None,'source_hashes':source_hashes}
    class Bind(ast.NodeTransformer):
        def visit_ImportFrom(self,n):
            n.names=[x for x in n.names if(x.asname or x.name)not in supplied]
            return n if n.names else None
        def visit_FunctionDef(self,n):
            if n.name in ('source_hashes','validate_output','preflight'):return None
            return self.generic_visit(n)
        def visit_Assign(self,n):
            if any(isinstance(x,ast.Name)and x.id=='PROTOCOL'for x in n.targets):
                n.value=ast.Constant(VERSION);return n
            return self.generic_visit(n)
        def visit_Constant(self,n):
            if isinstance(n.value,str):
                n.value=labels(n.value,b).replace('S1-01',instance)
                if n.value.startswith('frozen RPent tools + shared public'):
                    n.value='frozen RPent ordinary tools including VLA, no pick/place macro + own accepted S1 + persistent two-round owner bridge'
            return n
        def visit_Dict(self,n):
            n=self.generic_visit(n);keys=[x.value for x in n.keys if isinstance(x,ast.Constant)]
            if 'instructions'in keys:
                n.values[keys.index('instructions')]=ast.parse(repr(list(c.instructions)),mode='eval').body
                for key,value in [('own_accepted_engineering',b.acceptance),('native_binding_ready',False),('natural_method_ready',False)]:
                    n.keys.append(ast.Constant(key));n.values.append(ast.Constant(value))
            if 'source_count'in keys:
                n.keys.append(ast.Constant('instance'));n.values.append(ast.Constant(instance))
                n.keys.append(ast.Constant('natural_method_ready'));n.values.append(ast.Constant(False))
            return n
        def visit_Call(self,n):
            n=self.generic_visit(n)
            if isinstance(n.func,ast.Name)and n.func.id=='PanPickPlaceToolkit':ordinary.append('ordinary')
            return n
    tree=Bind().visit(tree)
    import os
    if not source_custody_enabled() and os.environ.get('SAFEVLA_LEASH_INSTANCE') == 'S1-01':
        from research.v4.bridge.run_b0 import (bind_s101_main, bind_lite_main,
                                              active_call_budgets, active_live_limits)
        supplied.update(active_call_budgets=active_call_budgets,
                        active_live_limits=active_live_limits)
        tree = bind_lite_main(bind_s101_main(tree), 'S1-01')
    ast.fix_missing_locations(tree)
    if ordinary!=['ordinary']:raise ValueError('v8 macro construction changed')
    name=__package__+'._runner_'+instance.replace('-','_');m=ModuleType(name)
    m.__file__=str(path);m.__package__=__package__;sys.modules[name]=m;m.__dict__.update(supplied)
    def validate_output(path,method,seed,*,runs=None):
        runs=Path(runs)if runs is not None else m.RUNS
        path=Path(path).resolve();prefix=instance.replace('-','_')
        if method not in {'B0','Ours'}or seed not in(0,1,2)or path.parent!=runs.resolve()or not re.fullmatch(rf'V4D\d{{4}}_{prefix}_{method}_smoke_seed{seed}',path.name):
            raise ValueError('fresh root-assigned own-instance development output required')
        if path.exists():raise FileExistsError(path)
        return path
    def preflight(a,*,runs=None):
        output=validate_output(a.output,a.method,a.seed,runs=runs)
        if a.egl_device<0 or a.physical_gpu<0:raise ValueError('physical GPU indexes required')
        m.validate_services(a)
        return output,source_hashes()
    m.validate_output=validate_output;m.preflight=preflight
    exec(compile(tree,str(path)+' [own S1 binding]','exec'),m.__dict__)
    # Core pins the same complete accepted/runtime/config union as this main.
    c.core.source_hashes=source_hashes
    if source_custody_enabled():
        m.compiled_main_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    return m
