"""Root-only P4 v3 native QA: 120 seconds, twenty neutral empty-gripper steps."""
from pathlib import Path
from unittest.mock import patch
import hashlib
from . import native_clear_qa
from .stand_scene import P4VisibleStandFacade

original = native_clear_qa.original
HERE = Path(__file__).resolve().parent
SOURCE = native_clear_qa.SOURCE + tuple(HERE/name for name in (
    'stand_scene.py', 'native_stand_qa.py', 'test_stand_cpu.py'))


def hashes():
    return {str(p.relative_to(original.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in SOURCE}


def main():
    with patch.object(original, 'Scene', P4VisibleStandFacade), \
            patch.object(original, 'HERE', HERE), patch.object(original, 'hashes', hashes):
        original.main()


if __name__ == '__main__':
    main()
