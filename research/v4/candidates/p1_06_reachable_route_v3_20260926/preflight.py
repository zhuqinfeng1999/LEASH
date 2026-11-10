import hashlib,json
from .scene import HERE,ROOT
from research.v4.candidates.p1_06_high_neck_v2_20260926.preflight import hashes as previous
def hashes():
 pins=previous()
 for p in [*HERE.glob('*.py'),HERE/'protocol.json']:pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 freeze=HERE/'source_freeze.json'
 if freeze.exists()and json.loads(freeze.read_text())!=pins:raise ValueError('nearer-bucket source drift')
 return pins
