"""Thin own bindings to original R3 gates and strict private evaluator."""
import ast,types
from pathlib import Path
from .config import binding,public_aliases
from .public import proposal
ROOT=Path(__file__).resolve().parents[4]

def gates(instance):
 b=binding(instance);path=ROOT/'research/v4/candidates/r3_v42_visibility_20260926/r3_gates.py';tree=ast.parse(path.read_text());counts={'aliases':0,'source':0,'proposal':0}
 class Own(ast.NodeTransformer):
  def visit_Assign(self,n):
   if any(isinstance(x,ast.Name)and x.id=='aliases'for x in n.targets):counts['aliases']+=1;n.value=ast.Name('OWN_ALIASES',ast.Load());return n
   return self.generic_visit(n)
  def visit_Constant(self,n):
   if n.value=='moka':counts['source']+=1;return ast.copy_location(ast.Constant(b.source),n)
   return n
  def visit_FunctionDef(self,n):
   if n.name=='_r3_proposal':
    counts['proposal']+=1
    n.body=ast.parse('return own_proposal(BINDING,name,arguments,workspace,locate_vla_target)').body;return n
   return self.generic_visit(n)
 tree=ast.fix_missing_locations(Own().visit(tree))
 if counts['aliases']!=2 or counts['proposal']!=1 or counts['source']<1:raise RuntimeError('R3 own gate source anchor drift')
 m=types.ModuleType(__name__+'._'+instance.replace('-','_'));m.__package__=__package__;m.__dict__.update(BINDING=b,OWN_ALIASES=public_aliases(b),own_proposal=proposal);exec(compile(tree,str(path)+'[own-public-role]','exec'),m.__dict__)
 return m

def strict_evaluator(instance):
 # Existing qualified derived scorer, only registered identity rebound.
 path=ROOT/'research/v4/candidates/r3_03_t1_derived_20260926/offline_evaluator.py'
 class Identity(ast.NodeTransformer):
  def visit_Constant(self,n):return ast.copy_location(ast.Constant(instance),n)if n.value=='R3-03'else n
 m=types.ModuleType(__name__+'._private_'+instance.replace('-','_'));exec(compile(ast.fix_missing_locations(Identity().visit(ast.parse(path.read_text()))),str(path)+'[private-own-identity]','exec'),m.__dict__)
 return m.Evaluator

