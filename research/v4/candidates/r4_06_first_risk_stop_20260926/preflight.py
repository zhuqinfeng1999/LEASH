"""Stopped bounded candidate; no qualification or simulator from preflight."""
import hashlib,json
from pathlib import Path
from research.v4.candidates.r4_crib_doll_collision_20260926 import preflight as parent
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def hashes():
 pins=parent.hashes()
 for p in[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'initial_geometry_seed0_fail.json']:
  pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 pins=dict(sorted(pins.items()));f=HERE/'source_freeze.json'
 if f.exists()and pins!=json.loads(f.read_text()):raise RuntimeError('Stopped first-risk wrapper source drift')
 return pins
if __name__=='__main__':print(json.dumps(dict(source_count=len(hashes()),models_called=[],gpu_used=False,qualification='STOPPED at seed0 initialgeometry',will_construct_scene=False)))
