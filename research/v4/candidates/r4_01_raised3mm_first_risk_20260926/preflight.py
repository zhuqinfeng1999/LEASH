import json,hashlib
from .scene import HERE,ROOT
from research.v4.candidates.r4_derived_first_risk_20260926 import preflight as parent
def hashes():
 pins=parent.hashes()
 paths=[*HERE.glob('*.py'),HERE/'protocol.json']+[ROOT/p for p in ['research/v4/accelerated/r4_native.py','research/v4/accelerated/r4_script_acceptance.py','research/v4/tasks/r4/template.py','research/v4/tasks/r4/scene.py','research/v4/tasks/r4/plastic_bag.xml','research/v4/tasks/r4/plastic_bag_print.png','research/v4/tasks/r4/assets.json','upstream/LIBERO/libero/libero/envs/env_wrapper.py']]
 for p in paths:pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 pins=dict(sorted(pins.items()));f=HERE/'source_freeze.json'
 if f.exists()and pins!=json.loads(f.read_text()):raise RuntimeError('R401 source drift')
 return pins
