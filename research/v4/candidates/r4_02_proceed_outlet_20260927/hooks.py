"""Only R402 generated original main, preserving services/finish/budgets."""
import ast,types,copy,hashlib,json
from pathlib import Path
from research.v4.candidates.r4_family_method_config_20260927 import hooks as family
from research.v4.accelerated.r4_adapter import R4V42Adapter
from .scene import Scene
from .gates import Adapter,Toolkit
from .feedback import SharedGripFeedback
from . import public
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
VERSION=family.VERSION+'-R402-action-commit-public-latch-v1'
def hashes():
 pins=json.loads((HERE/'source_freeze.json').read_text())['sha256'];now={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()for n in pins}
 if now!=pins:raise RuntimeError('R402 outlet source drift')
 return now
def adapter(scene,*,method,**kw):return Adapter(scene,**kw)if method=='Ours'else R4V42Adapter(scene,**kw)
def configure(bridge,adapter,decision,reviewer,sam_client,output):
 logs=[];holder={}
 def snapshot():
  t=holder['toolkit'];names=set(t.gate.task.constraints.forbidden_entities)
  if t.gate.lease:names.update(t.gate.lease.registered_entities)
  return public.workspace(adapter.scene,adapter,sam_client,logs,need_infant='infant'in names)
 t=Toolkit(bridge.base,adapter,decision=decision,reviewer=reviewer,workspace=snapshot,round_id='R4-02-current-round1');t.perception_log=logs;holder['toolkit']=t;bridge.base=t;return t
def runner_module():
 # Reuse family transformation rather than copying a runner/executor.
 path=ROOT/'research/v4/accelerated/run_r4.py';tree=ast.parse(path.read_text());counts=dict(scene=0,eval=0,evidence=0,hash=0,adapter=0,construct=0,start=0,proceed=0,protocol=0,shared=0)
 original_main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
 planner=next(n.orelse for n in ast.walk(original_main)if isinstance(n,ast.If)and ast.unparse(n.test)=="a.method == 'Ours'")
 class Own(ast.NodeTransformer):
  def visit_ImportFrom(self,n):
   repl={'research.v4.accelerated.r4_native':('scene','Scene=OWN_SCENE'),'research.v4.tasks.r4.evaluator':('eval','Evaluator=OWN_EVALUATOR'),'research.v4.accelerated.native_public':('evidence','evidence_from_scene=OWN_PUBLIC.task_evidence')}
   if n.module in repl:k,s=repl[n.module];counts[k]+=1;return ast.parse(s).body
   return n
  def visit_FunctionDef(self,n):
   if n.name=='hashes':counts['hash']+=1;n.body=ast.parse('return OWN_HASHES()').body;return n
   return self.generic_visit(n)
  def visit_Call(self,n):
   if isinstance(n.func,ast.Name)and n.func.id=='Scene':counts['construct']+=1;n.keywords.extend(ast.parse('f(camera_observations=True,render_gpu_device_id=a.egl_device)').body[0].value.keywords)
   if isinstance(n.func,ast.Name)and n.func.id=='R4V42Adapter':counts['adapter']+=1;n.func=ast.Name('OWN_ADAPTER',ast.Load());n.keywords.extend(ast.parse('f(method=a.method)').body[0].value.keywords)
   return self.generic_visit(n)
  def visit_Expr(self,n):
   if ast.unparse(n)=='reviewer.start()':counts['start']+=1;return [n,*ast.parse('adapter.bind_reviewer(reviewer)').body]
   return self.generic_visit(n)
  def visit_If(self,n):
   if ast.unparse(n.test)=="decision.result == 'PROCEED'":
    counts['proceed']+=1;n.body=ast.parse('OWN_CONFIGURE(bridge,adapter,decision,reviewer,Sam3Client(sam3),output)').body+copy.deepcopy(planner)
   return self.generic_visit(n)
  def visit_Assign(self,n):
   if any(isinstance(x,ast.Name)and x.id=='PROTOCOL'for x in n.targets):counts['protocol']+=1;n.value=ast.Constant(VERSION)
   if isinstance(n.value,ast.Tuple)and [getattr(x,'value',None)for x in n.value.elts]==['SAFE_SUCCESS','reason_judge_pending']:
    # A pending offline verdict must never be shown as accepted success.
    n.value.elts[0]=ast.Constant('SAFE_INCOMPLETE')
   n=self.generic_visit(n)
   if ast.unparse(n)=='base.episode_budget = 144':counts['shared']+=1;return[n,*ast.parse('base=OWN_FEEDBACK(base,scene)').body]
   return n
  def visit_Constant(self,n):
   if n.value=='R4-01':return ast.copy_location(ast.Constant('R4-02'),n)
   if n.value=='R4-01-seed':return ast.copy_location(ast.Constant('R4-02-seed'),n)
   return n
 tree=ast.fix_missing_locations(Own().visit(tree))
 if counts!=dict(scene=1,eval=1,evidence=1,hash=1,adapter=1,construct=1,start=1,proceed=1,protocol=1,shared=1):raise RuntimeError('Original R402 main anchors drift: '+repr(counts))
 main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main');tr=next(n for n in main.body if isinstance(n,ast.Try))
 tr.finalbody[:0]=ast.parse("if adapter is not None and hasattr(adapter,'commit_decisions'): report['commit_gate']=adapter.commit_decisions\nif bridge is not None:\n report['public_grip_sensor_feedback']=bridge.sensor_feedback\n if hasattr(bridge.base,'proposals'):\n  report['primitive_proposals']=bridge.base.proposals\n  report['action_gate']=[asdict(x)for x in bridge.base.gate.logs]\n  report['perception_log']=bridge.base.perception_log\nreport['public_sensor_revision']='R402-common-simulated-latch-v1'\nreport['tool_version']='RPent unchanged controls + shared current grip sensor feedback v1'").body
 tree=ast.fix_missing_locations(tree)
 from research.v4.candidates.r4_family_method_config_20260927.config import resolve,binding
 m=types.ModuleType(__package__+'._runner');m.__package__='research.v4.accelerated';m.__dict__.update(OWN_SCENE=Scene,OWN_EVALUATOR=resolve(binding('R4-02').evaluator),OWN_PUBLIC=public,OWN_HASHES=hashes,OWN_ADAPTER=adapter,OWN_CONFIGURE=configure,OWN_FEEDBACK=SharedGripFeedback)
 exec(compile(tree,str(path)+'[R402-proceed-outlet]','exec'),m.__dict__);return m
def finalizer_module():
 m=family.finalizer_module('R4-02');path=ROOT/'research/v4/accelerated/finalize_r4.py';tree=ast.parse(path.read_text())
 class Own(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value=='v4.2-dev-r4-assisted-640x480-48x144-80turn-wall1800':return ast.copy_location(ast.Constant(VERSION),n)
   if n.value=='R4-01':return ast.copy_location(ast.Constant('R4-02'),n)
   return n
 exec(compile(ast.fix_missing_locations(Own().visit(tree)),str(path)+'[R402-offline]','exec'),m.__dict__);return m
