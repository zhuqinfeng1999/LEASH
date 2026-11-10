"""Guarded original P1 injection recipe with one P105 carrier substitution."""
import ast,hashlib,json
from pathlib import Path
from research.v4.candidates.p1_own_plain_config_20260927 import hooks as original
from research.v4.candidates.p1_own_plain_config_v2_20260927 import hooks as v2
from .factory import components
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];PACKAGE='research.v4.candidates.p1_own_plain_config_v4_20260927'
VERSION='p1-own-six-common-ordinary-rpent-public-aabb-v42-candidate-v4-p105-method-carrier'
def source_hashes():
 v2.source_hashes();expected=json.loads((HERE/'source_freeze.json').read_text())['sha256'];current={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()for n in expected}
 if current!=expected:raise RuntimeError('P1 v4 source drift')
 return current

def runner_module(instance):
 source_hashes();text=Path(original.__file__).read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='runner_module');function=ast.get_source_segment(text,node)
 ns={**vars(original),'components':components,'PACKAGE':PACKAGE,'VERSION':VERSION}
 exec(compile(function,str(Path(original.__file__)),'exec'),ns);return ns['runner_module'](instance)
