import hashlib,json
from pathlib import Path
from research.v4.candidates.p1_05_held_increment_v2_20260926.preflight import hashes as previous_hashes
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def hashes():
 result=previous_hashes()
 proof=ROOT/'research/v4/candidates/p1_05_held_increment_v2_20260926/same_state_clearance_proof_seed0.json';result[str(proof.relative_to(ROOT))]=hashlib.sha256(proof.read_bytes()).hexdigest()
 for f in [*HERE.glob('*.py'),HERE/'protocol.json']:result[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
 return dict(sorted(result.items()))
