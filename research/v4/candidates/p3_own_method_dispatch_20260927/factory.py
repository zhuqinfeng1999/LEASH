"""Semantic-only compilation of the existing P3 code; no new executor."""
import ast,hashlib,re,sys,copy
from functools import lru_cache
from dataclasses import replace
from types import ModuleType,SimpleNamespace
from pathlib import Path
from .config import binding,ROOT
from research.v4.accelerated.p3_gated_adapter import P3GatedAdapter
from research.v4.accelerated.p3_offline_scoring import P3V42Evaluator
from research.v4.candidates.e1_02_plain_rpent_entry_20260927.factory import PlainRPentToolkit
PACKAGE='research.v4.candidates.p3_own_method_dispatch_20260927'
PATHS=dict(review='research/v4/accelerated/p3_review_names.py',
 held='research/v4/candidates/p3_held_sweep_20260925/held_sweep.py',
 toolkit='research/v4/accelerated/p3_gated_toolkit.py')

def relabel(value,b):
    words={'laptop computer':b.protected_prompt,'computer':b.protected_query2,
           'laptop':b.protected,'cup':b.source,'coaster':b.target}
    pattern=r'\b('+ '|'.join(re.escape(x)for x in sorted(words,key=len,reverse=True))+r')\b'
    return re.sub(pattern,lambda m:words[m.group()],value)

def algorithm_sha(tree):
    class Text(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str):node.value='TEXT'
            return node
        def visit_ImportFrom(self,node):return None
    return hashlib.sha256(ast.dump(Text().visit(copy.deepcopy(tree)),include_attributes=False).encode()).hexdigest()

def semantic_module(path,b,*,suffix,supplied=None,remove=()):
    tree=ast.parse((ROOT/path).read_text());changes=[]
    class Labels(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            node.names=[x for x in node.names if x.name not in remove]
            return node if node.names else None
        def visit_Constant(self,node):
            if isinstance(node.value,str):
                new=relabel(node.value,b)
                if new!=node.value:changes.append((node.value,new));node.value=new
            return node
    tree=Labels().visit(tree);ast.fix_missing_locations(tree)
    name=PACKAGE+'._'+b.instance.replace('-','_')+'_'+suffix
    m=ModuleType(name);m.__file__=str(ROOT/path);sys.modules[name]=m
    m.__dict__.update(supplied or {})
    exec(compile(tree,str(ROOT/path)+' [own semantic P3]', 'exec'),m.__dict__)
    from research.v4.leash_v43.run_manifest import source_custody_enabled
    if source_custody_enabled():
        m.semantic_changes=tuple(changes);m.original_source_sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        m.actual_compiled_algorithm_sha256=algorithm_sha(tree)
    return m

@lru_cache(maxsize=5)
def components(instance):
    b,instruction,original_scene=binding(instance)
    review=semantic_module(PATHS['review'],b,suffix='review')
    held=semantic_module(PATHS['held'],b,suffix='held',
        supplied={'PublicSweptRadius':review.PublicSweptRadius,'UnresolvedPublicHazard':review.UnresolvedPublicHazard},
        remove=('PublicSweptRadius','UnresolvedPublicHazard'))
    toolkit=semantic_module(PATHS['toolkit'],b,suffix='toolkit',
        supplied={'PublicSweptRadius':review.PublicSweptRadius,'UnresolvedPublicHazard':review.UnresolvedPublicHazard,
                  'HeldCupSweep':held.HeldCupSweep},
        remove=('PublicSweptRadius','UnresolvedPublicHazard','HeldCupSweep'))
    # This adds task text metadata only; exact accepted constructor/physics remains.
    class Scene(original_scene):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);self.instruction=instruction
    Scene.__name__='Own'+instance.replace('-','')+'Scene'
    def normalize(**kwargs):
        result=review.normalize_p3_task_review(**kwargs)
        allowed=frozenset(b.source if x.lower()in b.source_aliases else x
            for x in result.decision.constraints.allowed_operated_entities)
        return replace(result,decision=replace(result.decision,
            constraints=replace(result.decision.constraints,allowed_operated_entities=allowed)))
    def locate(prompt,public):
        text=prompt.lower()
        for name,aliases in ((b.target,(b.target,)),(b.source,b.source_aliases)):
            box=public.box(name)
            if box is not None and any(re.search(r'\b'+re.escape(x)+r'\b',text)for x in aliases):
                return name,tuple((lo+hi)/2 for lo,hi in zip(box.lower,box.upper))
        return None
    guidance=review.P3_REVIEW_NAME_GUIDANCE+(
        '\nOutput vocabulary for this user instruction: allowed operated entity '+b.source+
        '; authorized target '+b.target+'. These names are formatting aliases, not evidence '+
        'of presence, pose, contents or safety. Identify them from the supplied public images.\n')
    return SimpleNamespace(binding=b,INSTRUCTION=instruction,Scene=Scene,AcceptedScene=original_scene,
        Evaluator=P3V42Evaluator,Adapter=P3GatedAdapter,PlainRPentToolkit=PlainRPentToolkit,
        review=review,held=held,toolkit=toolkit,normalize=normalize,locate=locate,guidance=guidance,
        natural_method_ready=False,native_binding_ready=False)
