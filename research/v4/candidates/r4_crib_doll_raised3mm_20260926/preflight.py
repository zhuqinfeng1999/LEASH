import hashlib,json
from pathlib import Path
from research.v4.candidates.r4_crib_doll_collision_20260926 import preflight as parent
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def hashes():
 pins=parent.hashes()
 for p in[*HERE.glob('*.py'),HERE/'crib_doll_raised3mm.xml',HERE/'protocol.json',HERE/'geometry10.json',HERE/'invariants.json']:
  pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 pins=dict(sorted(pins.items()));f=HERE/'source_freeze.json'
 if f.exists()and pins!=json.loads(f.read_text()):raise RuntimeError('Raised matching doll source drift')
 return pins
