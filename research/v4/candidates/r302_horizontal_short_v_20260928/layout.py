"""Single length from actual own hard/pad envelope; no pose/length search."""
import json
from pathlib import Path
import numpy as np

H=Path(__file__).resolve().parent
ORIGINAL=H.parent/'r302_horizontal_end_grasp_saved_copy_20260928'
geometry=json.loads((ORIGINAL/'hand_geometry.json').read_text())
ends=[]
for stage,rows in geometry.items():
    for r in rows:
        if 'vertices_eef' in r:
            v=np.asarray(r['vertices_eef'],float)
        else:
            # Actual compiled pad box corners, not an invented object AABB.
            signs=np.array([[x,y,z] for x in(-1,1) for y in(-1,1) for z in(-1,1)])
            v=(signs*np.asarray(r['size']))@np.asarray(r['local_rotation']).T+np.asarray(r['local_position'])
        # R_eef=[canY,canX,-canZ], zone axial coordinate +42 mm.
        ends.append((float(np.min(.042-v[:,2])),stage,r['name']))
MINIMUM_HAND_AXIAL,BOUNDING_STAGE,BOUNDING_GEOM=min(ends)
HARD_CLEARANCE=.003
HALF_LENGTH=MINIMUM_HAND_AXIAL-HARD_CLEARANCE
if not .015<HALF_LENGTH<.075:
    raise RuntimeError('No normal central support derived from exact hand envelope')
