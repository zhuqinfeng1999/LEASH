"""Only parser correction to the frozen zero-control higher-seat source."""
from pathlib import Path
SOURCE=Path(__file__).with_name('scene_v3.py').read_text()
BAD="name=='person_chair'else2"
if SOURCE.count(BAD)!=1:raise ValueError('Exact original syntax typo changed')
CORRECTED=SOURCE.replace(BAD,"name=='person_chair' else 2")
exec(compile(CORRECTED,__file__,'exec'),globals())
