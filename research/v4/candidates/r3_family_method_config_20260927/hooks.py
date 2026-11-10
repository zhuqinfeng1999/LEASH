"""Guarded, scoped own-role configuration of the existing R3 runner source.

No full runner copy, common-file mutation, model launch or new executor.
"""
import ast,types,hashlib,json
from pathlib import Path
from .config import binding,resolve,hazard_description
from .components import gates,strict_evaluator
from . import public
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
VERSION='v4.2-dev-r3-assisted-640x480-48x144-80turn-wall1800-family-own-public-envelope-v1'

def hashes():
 path=HERE/'source_freeze.json'
 if not path.exists():raise RuntimeError('family source must be frozen before dispatch')
 expected=json.loads(path.read_text())['sha256'];out={n:hashlib.sha256((Path(n)if n.startswith('/')else ROOT/n).read_bytes()).hexdigest()for n in expected}
 if out!=expected:raise RuntimeError('own R3 family source drift')
 return out

def entry_module(instance):
 b=binding(instance);g=gates(instance);path=ROOT/'research/v4/candidates/r3_v42_visibility_20260926/method_entry.py';tree=ast.parse(path.read_text());counts={'gates':0,'scene':0,'eval':0,'id':0}
 class Own(ast.NodeTransformer):
  def visit_ImportFrom(self,n):
   if n.module=='scene_lower_aim':counts['scene']+=1;return ast.parse('LowerAimR3Scene=OWN_SCENE').body
   if n.module=='r3_gates':
    counts['gates']+=1;return ast.parse('\n'.join(x.name+'=OWN_GATES.'+x.name for x in n.names)).body
   if n.module=='research.v4.tasks.r3.evaluator':counts['eval']+=1;return ast.parse('Evaluator=OWN_EVALUATOR').body
   return n
  def visit_Assign(self,n):
   ids={x.id for x in n.targets if isinstance(x,ast.Name)}
   if 'INSTANCE'in ids:counts['id']+=1;n.value=ast.Constant(instance)
   if 'HAZARD_DESCRIPTION'in ids:n.value=ast.Constant(hazard_description(b))
   return self.generic_visit(n)
 tree=ast.fix_missing_locations(Own().visit(tree))
 if counts!={'gates':1,'scene':1,'eval':1,'id':1}:raise RuntimeError('R3 entry anchor changed')
 m=types.ModuleType(__package__+'._entry_'+instance.replace('-','_'));m.__package__='research.v4.candidates.r3_v42_visibility_20260926';m.__dict__.update(OWN_SCENE=resolve(b.factory),OWN_GATES=g,OWN_EVALUATOR=strict_evaluator(instance));__import__('sys').modules[m.__name__]=m
 exec(compile(tree,str(path)+'[own-family-entry]','exec'),m.__dict__);return m

def runner_module(instance):
 b=binding(instance);e=entry_module(instance);g=gates(instance);path=ROOT/'research/v4/candidates/r3_v42_visibility_20260926/run_r3.py';tree=ast.parse(path.read_text());counts={'entry':0,'workspace':0,'target':0,'binder':0,'protocol':0,'hash':0}
 class Own(ast.NodeTransformer):
  def visit_ImportFrom(self,n):
   if n.module=='method_entry':counts['entry']+=1;return ast.parse('\n'.join(x.name+'=OWN_ENTRY.'+x.name for x in n.names)).body
   if n.module=='r3_gates':counts['binder']+=1;return ast.parse('bind_r3_task_names=OWN_GATES.bind_r3_task_names').body
   return n
  def visit_FunctionDef(self,n):
   if n.name=='hashes':counts['hash']+=1;n.body=ast.parse('return OWN_HASHES()').body;return n
   if n.name=='workspace':counts['workspace']+=1;n.body=ast.parse('return own_public.workspace(BINDING,entry,sam_client,perception_log)').body;return n
   if n.name=='locate_vla_target':counts['target']+=1;n.body=ast.parse('return own_public.locate(BINDING,prompt,public)').body;return n
   return self.generic_visit(n)
  def visit_Assign(self,n):
   if any(isinstance(x,ast.Name)and x.id=='PROTOCOL'for x in n.targets):counts['protocol']+=1;n.value=ast.Constant(VERSION)
   return self.generic_visit(n)
  def visit_Constant(self,n):return ast.copy_location(ast.Constant(b.source),n)if n.value=='moka'else n
 tree=ast.fix_missing_locations(Own().visit(tree))
 if counts!={'entry':1,'workspace':1,'target':1,'binder':1,'protocol':1,'hash':1}:raise RuntimeError('R3 runner own-role anchors drift: '+repr(counts))
 m=types.ModuleType(__package__+'._runner_'+instance.replace('-','_'));m.__package__='research.v4.candidates.r3_v42_visibility_20260926';m.__dict__.update(OWN_ENTRY=e,OWN_GATES=g,BINDING=b,OWN_HASHES=hashes,own_public=public);exec(compile(tree,str(path)+'[own-family-runner]','exec'),m.__dict__)
 return m
