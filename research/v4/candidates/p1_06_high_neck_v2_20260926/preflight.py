import hashlib,json
from pathlib import Path
from research.v4.candidates.p1_06_wine_bucket_t2_20260926.preflight import hashes as old_hashes
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def hashes():
 result=old_hashes()
 for p in[*HERE.glob('*.py'),HERE/'protocol.json']:result[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 result=dict(sorted(result.items()));freeze=HERE/'source_freeze.json'
 if freeze.exists()and json.loads(freeze.read_text())!=result:raise ValueError('P1-06 v2 source drift')
 return result
