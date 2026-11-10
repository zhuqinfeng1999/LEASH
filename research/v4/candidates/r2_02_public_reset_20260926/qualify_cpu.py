"""Thin scoped reuse of accepted scripts; fresh reset qualification only."""
import hashlib
from pathlib import Path
from unittest.mock import patch
from research.v4.candidates.r2_02_t1_derived_20260926 import scripts,qualify
from .scene import R2IncensePublicResetScene,HERE
def hashes():
    result=scripts.hashes();root=scripts.ROOT
    for path in list(HERE.glob('*.py'))+[HERE/'selected_pose.json']:
        result[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result
if __name__=='__main__':
    with patch.object(scripts,'T1R2DerivedScene',R2IncensePublicResetScene),patch.object(qualify,'HERE',HERE),patch.object(qualify,'hashes',hashes):
        raise SystemExit(qualify.main())
