"""Pin only four affected references and one independent collider candidate."""
import hashlib,json
from pathlib import Path
from research.v4.candidates.r4_02_plug_surface_v2_20260926 import qualify_cpu as cable
from research.v4.candidates.r4_05_identity_uv_fix_20260926 import preflight as beads
from research.v4.candidates.r4_06_flat_underbase_v2_20260926 import qualify_cpu as coins
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def hashes():
 pins={}
 for provider in(cable,beads,coins):
  current=provider.hashes()
  for k,v in current.items():
   if k in pins and pins[k]!=v:raise RuntimeError('Affected provider source disagreement')
   pins[k]=v
 paths=[*HERE.glob('*.py'),HERE/'crib_with_doll_collisions.xml',HERE/'protocol.json',HERE/'old_compiled_trajectory_audit.json',HERE/'initial_and_one_naive.json',HERE/'r401_saved_source_reference.json',ROOT/'research/v4/runs/V4D0355_R4_B0_flatbagprint_v3_seed0/scene.xml',ROOT/'research/v4/runs/V4D0355_R4_B0_flatbagprint_v3_seed0/source-sha256.json',ROOT/'research/v4/accelerated/r4_native.py']
 for p in paths:pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 pins=dict(sorted(pins.items()));f=HERE/'source_freeze.json'
 if f.exists()and pins!=json.loads(f.read_text()):raise RuntimeError('One-shot doll collider candidate source drift')
 return pins
if __name__=='__main__':print(json.dumps(dict(source_count=len(hashes()),models_called=[],gpu_used=False,current_acceptance=False,run_qualification=False)))
