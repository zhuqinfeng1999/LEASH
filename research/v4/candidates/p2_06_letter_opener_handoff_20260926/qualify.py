"""Existing three-family qualification, bounded seeds0..9; no model/GPU."""
import argparse,hashlib,json
from unittest.mock import patch
from research.v4.candidates.p2_03_screwdriver_handoff_20260926 import qualify as physical
from research.v4.candidates.p2_02_scissors_handoff_20260926 import qualify as loop
from .scene import HERE,ROOT
from . import scripts
def hashes():
 s=physical.hashes()
 for p in [*HERE.glob('*.py'),HERE/'protocol.json',HERE/'asset_meta.json',ROOT/'research/v4/candidates/p2_03_visible_screwdriver_20260926/scene.py',*sorted((HERE/'assets').glob('*'))]:
  if p.is_file():s[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 f=HERE/'source_freeze.json'
 if f.exists()and s!=json.loads(f.read_text()):raise ValueError('P206 source drift')
 return s
def main():
 source=hashes()
 with patch.multiple(loop,HERE=HERE,scripts=scripts,hashes=hashes):code=loop.main()
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=3);a=p.parse_args();f=HERE/f'qualification_seed0to{a.seeds-1}.json';r=json.loads(f.read_text());r['instance']='P2-06';f.write_text(json.dumps(r,indent=2)+'\n');return code
if __name__=='__main__':raise SystemExit(main())
