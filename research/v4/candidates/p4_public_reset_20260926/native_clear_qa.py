"""P4 selected-reset root QA; same120s/20-neutral-step native procedure."""
from pathlib import Path
from unittest.mock import patch
import hashlib
from research.v4.candidates.p4_v42_t4_tabletop_20260926 import native_visual as original
from .clear_scene import P4ClearViewFacade

HERE=Path(__file__).resolve().parent
SOURCE=tuple(original.SOURCE)+tuple(HERE/name for name in (
    'scene.py','selected_pose.json','clear_scene.py','selected_pose_v2.json',
    'native_clear_qa.py','test_cpu.py','test_clear_cpu.py'))

def hashes():
    return {str(p.relative_to(original.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE}

def main():
    with patch.object(original,'Scene',P4ClearViewFacade),patch.object(original,'HERE',HERE),patch.object(original,'hashes',hashes):
        original.main()

if __name__=='__main__':main()
