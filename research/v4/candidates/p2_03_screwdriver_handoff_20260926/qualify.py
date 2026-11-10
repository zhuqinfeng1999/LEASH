"""Existing bounded three-family loop; only screwdriver configuration selected."""
import hashlib,json
from unittest.mock import patch
from research.v4.candidates.p2_02_scissors_handoff_20260926 import qualify as loop
from research.v4.candidates.e3_02_teapot_teacup_20260926 import qualify as common
from .scene import HERE,ROOT
from . import scripts
def hashes():
 s=common.hashes();names=['scene_safe_slot_v2.py','evaluator.py','scripted_safe_slot_v2.py','native_visual_qa_blade_side_v3.py']
 paths=[ROOT/'research/v4/candidates/p2_v42_t3_20260925'/n for n in names]
 paths+=[ROOT/'research/v3/tasks/p2/evaluator.py',ROOT/'research/v3/tasks/p2/scene.py',ROOT/'research/v4/accelerated/scoring.py',ROOT/'research/v4/accelerated/slot_preflight.py',ROOT/'research/upstream/polyhaven/wood_table_diff_1k.png',ROOT/'research/v4/candidates/p2_02_scissors_handoff_20260926/qualify.py']
 paths+=[*HERE.glob('*.py'),HERE/'protocol.json',ROOT/'research/v4/candidates/e1_03_screwdriver_toolbox_20260926/assets/screwdriver.xml',ROOT/'research/v4/candidates/e1_03_screwdriver_toolbox_20260926/asset_meta.json']
 for p in paths:s[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 f=HERE/'source_freeze.json'
 if f.exists()and s!=json.loads(f.read_text()):raise ValueError('P203 source drift')
 return s
def main():
 source=hashes()
 with patch.multiple(loop,HERE=HERE,scripts=scripts,hashes=hashes):code=loop.main()
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=3);a=p.parse_args();f=HERE/f'qualification_seed0to{a.seeds-1}.json';r=json.loads(f.read_text());r['instance']='P2-03';f.write_text(json.dumps(r,indent=2)+'\n');return code
if __name__=='__main__':raise SystemExit(main())
