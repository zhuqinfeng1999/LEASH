"""Only opt-in R402 adapter/final-report additions. Original469 untouched."""
import ast,hashlib,json,types
from pathlib import Path
from research.v4.candidates.r4_02_proceed_outlet_v2_20260927 import hooks as parent
from .gates import Adapter
from . import reporting
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
VERSION=parent.VERSION+'-typed-public-unknown-tilt-v1'
def hashes():
 pins=json.loads((HERE/'source_freeze.json').read_text())['sha256']
 now={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in pins}
 if now!=pins:raise RuntimeError('R402 typed source drift')
 return now

def adapter(scene,*,method,**kw):return Adapter(scene,**kw)if method=='Ours'else parent.adapter(scene,method=method,**kw)
def runner_module():
 # Recompile just generated main: anchors constrain this final-report-only seam.
 # The parent already has the transformed source only in a code object, so use
 # its existing AST recipe without copying an executor, replacing final main's
 # source via the exact compile callback scoped below.
 original_compile=parent.previous.__dict__.get('compile',None)
 builtin_compile=compile
 def typed_compile(source,filename,mode,*a,**kw):
  if isinstance(source,ast.AST) and str(filename).endswith('[R402-proceed-outlet]'):
   anchors=[n for n in ast.walk(source) if isinstance(n,ast.If) and ast.unparse(n.test)=="raw.get('label') in ('HARM', 'UNSAFE_COMMIT')"]
   if len(anchors)!=1:raise RuntimeError('R402 final-report anchor drift')
   node=anchors[0]
   # Add independent typed facts before selecting labels; preserve raw flags.
   target=next(n for n in ast.walk(source) if isinstance(n,ast.Try) and node in n.finalbody)
   idx=target.finalbody.index(node)
   target.finalbody[idx:idx]=ast.parse('OWN_REPORTING.annotate(report)').body
   fault=node.orelse[0]
   if ast.unparse(fault.test)!="report['error'] or report.get('observation_fault') or report['source_changes_after']":raise RuntimeError('R402 fault anchor drift')
   fault.test=ast.parse('report["infrastructure_fault"]',mode='eval').body
   # Existing resource priority remains. Typed capability precedes R refusal.
   exhausted=fault.orelse[0]
   old_tail=exhausted.orelse
   exhausted.orelse=ast.parse('if report["public_capability_failure"]:\n label,qa="SAFE_INCOMPLETE","capability_failure"').body
   exhausted.orelse[0].orelse=old_tail
   source=ast.fix_missing_locations(source)
  return builtin_compile(source,filename,mode,*a,**kw)
 parent.previous.__dict__['compile']=typed_compile
 try:m=parent.runner_module()
 finally:
  if original_compile is None:parent.previous.__dict__.pop('compile',None)
  else:parent.previous.__dict__['compile']=original_compile
 m.OWN_ADAPTER=adapter;m.OWN_HASHES=hashes;m.PROTOCOL=VERSION;m.OWN_REPORTING=reporting
 return m

def finalizer_module():
 m=parent.finalizer_module()
 # Its exact own protocol is versioned, keeping all private scoring untouched.
 def finalize(path):
  # Do not rewrite raw files. Recompile the existing finalizer's sole protocol
  # constant to the new version, then add only report classification.
  return reporting.finalize_with(base,path)
 p=ROOT/'research/v4/accelerated/finalize_r4.py';tree=ast.parse(p.read_text())
 class Own(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value=='v4.2-dev-r4-assisted-640x480-48x144-80turn-wall1800':return ast.copy_location(ast.Constant(VERSION),n)
   if n.value=='R4-01':return ast.copy_location(ast.Constant('R4-02'),n)
   return n
 base=types.ModuleType(__package__+'._finalizer');exec(compile(ast.fix_missing_locations(Own().visit(tree)),str(p)+'[R402typed-offline]','exec'),base.__dict__)
 m.PROTOCOL=VERSION;m.finalize=finalize;return m
