import hashlib,json
from pathlib import Path
from research.v4.candidates.p1_05_dumbbell_basket_t2_20260926.preflight import hashes as original_hashes
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def hashes():
 result=original_hashes()
 for f in [*HERE.glob('*.py'),HERE/'protocol.json']:result[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
 return dict(sorted(result.items()))
