"""One opt-in generated-main adapter substitution; original v1 stays frozen."""
import ast,hashlib,json
from pathlib import Path
from research.v4.candidates.r4_02_proceed_outlet_20260927 import hooks as previous
from .gates import Adapter
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
VERSION=previous.VERSION+'-held-axis-unknown-v2'
def hashes():
 pins=json.loads((HERE/'source_freeze.json').read_text())['sha256']
 now={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()for n in pins}
 if now!=pins:raise RuntimeError('R402 v2 source drift')
 return now
def adapter(scene,*,method,**kw):return Adapter(scene,**kw)if method=='Ours'else previous.adapter(scene,method=method,**kw)
def runner_module():
 m=previous.runner_module();m.OWN_ADAPTER=adapter;m.OWN_HASHES=hashes;m.PROTOCOL=VERSION;return m

def finalizer_module():
 m=previous.finalizer_module();path=ROOT/'research/v4/accelerated/finalize_r4.py';tree=ast.parse(path.read_text())
 class Own(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value=='v4.2-dev-r4-assisted-640x480-48x144-80turn-wall1800':return ast.copy_location(ast.Constant(VERSION),n)
   if n.value=='R4-01':return ast.copy_location(ast.Constant('R4-02'),n)
   return n
 exec(compile(ast.fix_missing_locations(Own().visit(tree)),str(path)+'[R402-v2-offline]', 'exec'),m.__dict__);return m
