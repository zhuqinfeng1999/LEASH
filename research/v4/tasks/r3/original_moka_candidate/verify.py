#!/usr/bin/env python3
"""CPU-only integrity, size and geometry checks for this source candidate."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

root=Path(__file__).resolve().parent
out=root/'generated'
hashes=json.loads((out/'sha256.json').read_text())
for name,want in hashes.items():
    assert hashlib.sha256((out/name).read_bytes()).hexdigest()==want,name
for name,size in {
    'brushed_aluminium_albedo_2k.png':(2048,2048),
    'brushed_aluminium_roughness_2k.png':(2048,2048),
    'satin_grip_albedo_2k.png':(2048,2048),
    'satin_grip_roughness_2k.png':(2048,2048),
    'neutral_microwave_metal_1k.png':(1024,1024),
}.items():
    with Image.open(out/name) as im: assert im.size==size,(name,im.size)
verts=[]
for path in sorted(out.glob('*.obj')):
    for line in path.read_text().splitlines():
        if line.startswith('v '): verts.append([float(x) for x in line.split()[1:]])
arr=np.asarray(verts)
assert len(arr)>2000
bbox=[arr.min(axis=0),arr.max(axis=0)]
height=float(bbox[1][2]-bbox[0][2])
assert .151<=height<=.153,height
assert abs(float(bbox[1][1]))<.090
assert abs(float(bbox[0][1]))<.060
collision=json.loads((out/'collision_proxy.json').read_text())
assert collision['collision_height_m']>.144
assert all(collision['tilt_fit'].values())
assert set(hashes)=={p.name for p in out.iterdir() if p.is_file() and p.name!='sha256.json'}
print(json.dumps({'status':'CPU_CANDIDATE_VALID','files':len(hashes),
                  'visual_vertices':len(verts),'visual_bbox_min_m':bbox[0].tolist(),
                  'visual_bbox_max_m':bbox[1].tolist(),
                  'visual_height_m':height,'tilt_fit':collision['tilt_fit']},indent=2))
