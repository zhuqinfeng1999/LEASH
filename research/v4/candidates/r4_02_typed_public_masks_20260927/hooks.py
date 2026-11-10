"""Opt-in public workspace + reporter bindings, not an executor replacement."""
import ast,types,json,hashlib
from pathlib import Path
from research.v4.candidates.r4_02_unknown_tilt_typed_20260927 import hooks as parent
from research.v4.candidates.r4_02_proceed_outlet_20260927 import hooks as original
from . import public,reporting
from .gates import Toolkit
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
VERSION=parent.VERSION+'-typed-public-SAM-missing-v1'
def hashes():
 p=json.loads((HERE/'source_freeze.json').read_text())['sha256'];now={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()for n in p}
 if now!=p:raise RuntimeError('R402 typed public source drift')
 return now

def configure(*args,**kw):
 # Rebind the existing tiny configure function's globals in an isolated namespace.
 # No mutable shared module monkeypatch survives success or constructor faults.
 f=original.configure
 namespace={**f.__globals__,'Toolkit':Toolkit,'public':types.SimpleNamespace(workspace=public.workspace)}
 return types.FunctionType(f.__code__,namespace,f.__name__,f.__defaults__,f.__closure__)(*args,**kw)

def runner_module():
 m=parent.runner_module();m.OWN_CONFIGURE=configure;m.OWN_REPORTING=reporting;m.OWN_HASHES=hashes;m.PROTOCOL=VERSION;return m

def finalizer_module():
 p=ROOT/'research/v4/accelerated/finalize_r4.py';tree=ast.parse(p.read_text())
 class Own(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value=='v4.2-dev-r4-assisted-640x480-48x144-80turn-wall1800':return ast.copy_location(ast.Constant(VERSION),n)
   if n.value=='R4-01':return ast.copy_location(ast.Constant('R4-02'),n)
   return n
 base=types.ModuleType(__package__+'._offline');exec(compile(ast.fix_missing_locations(Own().visit(tree)),str(p)+'[R402typedSAM-offline]','exec'),base.__dict__)
 m=types.ModuleType(__package__+'._finalizer');m.finalize=lambda path:reporting.finalize_with(base,path);return m
