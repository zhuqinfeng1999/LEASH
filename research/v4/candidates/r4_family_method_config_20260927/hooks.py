"""Only guarded existing runner/finalizer configuration; NO new executor."""
import ast,types,hashlib,json
from pathlib import Path
from .config import ROOT,HERE,binding,resolve,scene_class
VERSION='v4.2-dev-r4-assisted-640x480-48x144-80turn-wall1800-own-physical-family-v1'
def hashes():
 expected=json.loads((HERE/'source_freeze.json').read_text())['sha256']
 actual={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()for n in expected}
 if actual!=expected:raise RuntimeError('R4 family source drift')
 return actual
def runner_module(instance):
 b=binding(instance);path=ROOT/'research/v4/accelerated/run_r4.py';tree=ast.parse(path.read_text());counts=dict(scene=0,eval=0,hash=0,construct=0,protocol=0)
 class Own(ast.NodeTransformer):
  def visit_ImportFrom(self,n):
   if n.module=='research.v4.accelerated.r4_native':counts['scene']+=1;return ast.parse('Scene=OWN_SCENE').body
   if n.module=='research.v4.tasks.r4.evaluator':counts['eval']+=1;return ast.parse('Evaluator=OWN_EVALUATOR').body
   return n
  def visit_FunctionDef(self,n):
   if n.name=='hashes':counts['hash']+=1;n.body=ast.parse('return OWN_HASHES()').body;return n
   return self.generic_visit(n)
  def visit_Call(self,n):
   if isinstance(n.func,ast.Name)and n.func.id=='Scene':
    counts['construct']+=1;n.keywords.extend([ast.keyword(arg='camera_observations',value=ast.Constant(True)),ast.keyword(arg='render_gpu_device_id',value=ast.Attribute(ast.Name('a',ast.Load()),'egl_device',ast.Load()))])
   return self.generic_visit(n)
  def visit_Assign(self,n):
   if any(isinstance(x,ast.Name)and x.id=='PROTOCOL'for x in n.targets):counts['protocol']+=1;n.value=ast.Constant(VERSION)
   return self.generic_visit(n)
  def visit_Constant(self,n):
   if n.value=='R4-01':return ast.copy_location(ast.Constant(instance),n)
   if n.value=='R4-01-seed':return ast.copy_location(ast.Constant(instance+'-seed'),n)
   return n
 tree=ast.fix_missing_locations(Own().visit(tree))
 if counts!=dict(scene=1,eval=1,hash=1,construct=1,protocol=1):raise RuntimeError('Original R4 runner anchors changed: '+repr(counts))
 m=types.ModuleType(__package__+'._runner_'+instance.replace('-','_'));m.__package__='research.v4.accelerated';m.__dict__.update(OWN_SCENE=scene_class(instance),OWN_EVALUATOR=resolve(b.evaluator),OWN_HASHES=hashes)
 exec(compile(tree,str(path)+'[own-R4-family]','exec'),m.__dict__)
 return m
def finalizer_module(instance):
 b=binding(instance);path=ROOT/'research/v4/accelerated/finalize_r4.py';tree=ast.parse(path.read_text());counts=0
 class Own(ast.NodeTransformer):
  def visit_Constant(self,n):
   nonlocal counts
   if n.value=='v4.2-dev-r4-assisted-640x480-48x144-80turn-wall1800':counts+=1;return ast.copy_location(ast.Constant(VERSION),n)
   if n.value=='R4-01':return ast.copy_location(ast.Constant(instance),n)
   return n
 tree=ast.fix_missing_locations(Own().visit(tree))
 if counts!=1:raise RuntimeError('Original offline finalizer protocol anchor changed')
 m=types.ModuleType(__package__+'._finalizer_'+instance.replace('-','_'));exec(compile(tree,str(path)+'[own-R4-family]','exec'),m.__dict__);return m
def judge_module(instance):
 """Own hazard goes only to the ORIGINAL offline, method-blind judge."""
 b=binding(instance);path=ROOT/'research/v4/accelerated/reason_judge.py';tree=ast.parse(path.read_text());count=0
 class Own(ast.NodeTransformer):
  def visit_Assign(self,n):
   nonlocal count
   if any(isinstance(x,ast.Name)and x.id=='HAZARDS'for x in n.targets):
    count+=1;n.value=ast.Dict(keys=[ast.Constant(instance)],values=[ast.Constant(b.hazard)])
   return n
 tree=ast.fix_missing_locations(Own().visit(tree))
 if count!=1:raise RuntimeError('Original offline judge hazard anchor changed')
 m=types.ModuleType(__package__+'._judge_'+instance.replace('-','_'));exec(compile(tree,str(path)+'[own-R4-offline-judge]','exec'),m.__dict__);return m
