import hashlib,json
from unittest.mock import patch
from .scripts import HERE,ROOT,run_case
from research.v4.candidates.r3_06_mouth_frame_20260926 import qualify as parent
OLD_HASHES=parent.hashes;OLD_HERE=parent.HERE
def hashes():
 with patch.object(parent,'HERE',OLD_HERE):out=OLD_HASHES()
 for p in [*HERE.glob('*.py'),HERE/'HYPOTHESIS.md',ROOT/'research/v4/candidates/r4_05_live_rim_grasp_20260926/native_motion.py',ROOT/'research/v4/candidates/scene_review_export_20260926/recorder.py']:
  out[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 f=HERE/'source_freeze.json'
 if f.exists()and out!=json.loads(f.read_text()):raise ValueError('side55 source drift')
 return out
def main():
 with patch.object(parent,'HERE',HERE),patch.object(parent,'hashes',hashes),patch.object(parent,'run_case',run_case):return parent.main()
if __name__=='__main__':raise SystemExit(main())
