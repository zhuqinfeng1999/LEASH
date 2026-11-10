import hashlib,json
from pathlib import Path
from research.v4.candidates.r4_crib_doll_raised3mm_20260926 import preflight as pose
from research.v4.candidates.r4_06_first_risk_stop_20260926 import preflight as stopped
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def hashes():
 pins={**pose.hashes(),**stopped.hashes()}
 for p in[*HERE.glob('*.py'),HERE/'protocol.json']:
  pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 pins=dict(sorted(pins.items()));f=HERE/'source_freeze.json'
 if f.exists()and pins!=json.loads(f.read_text()):raise RuntimeError('R406 raised-first-risk source drift')
 return pins
