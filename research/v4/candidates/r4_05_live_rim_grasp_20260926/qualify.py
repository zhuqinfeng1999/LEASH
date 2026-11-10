"""Thin parent batch wrapper; same physical Scene/evaluator, changed offline approach."""
import hashlib,json
from pathlib import Path
from unittest.mock import patch
from .scripts import run_case
from research.v4.candidates.r4_05_inclined_identity_v2_20260926 import qualify_cpu as parent
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
ORIGINAL_HASHES=parent.hashes
ORIGINAL_HERE=parent.HERE
def hashes():
    with patch.object(parent,'HERE',ORIGINAL_HERE):pins=ORIGINAL_HASHES()
    for p in [*HERE.glob('*.py'),HERE/'HYPOTHESIS.md']:
        pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    f=HERE/'source_freeze.json'
    if f.exists()and pins!=json.loads(f.read_text()):raise ValueError('candidate source drift')
    return pins
def main():
    with patch.object(parent,'HERE',HERE),patch.object(parent,'run_case',run_case),patch.object(parent,'hashes',hashes):return parent.main()
if __name__=='__main__':raise SystemExit(main())
