"""P1 v5 clearance candidate: two natural extended arms, wider hand gap.

This isolated development scene keeps the same knife, plate, grasp assistance,
path hazard evaluator, perturbation seeds, and instructions as P1 v4. It only
moves each intact person's hand outward along the tabletop so the sharp knife
starts with visible clearance. No prior result is overwritten.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from unittest.mock import patch
from . import candidate_v4 as v4

class P1V5Facade(v4.P1V4Facade):
    e1_hand_start_xyz_m = (.10, .08, .938)
    e1_left_rest_xyz_m = (.20, -.44, .938)

P1V5Evaluator = v4.P1V4Evaluator

def probe(mode: str, seed: int):
    with patch.object(v4, 'P1V4Facade', P1V5Facade):
        row = v4.probe(mode, seed)
    row.update(kind='p1_v5_wide_clearance_cpu_candidate_not_native_or_method',
               scene_version='p1_v5_hand_y_plus080_left_y_minus440')
    return row

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('safe','naive','failure'))
    p.add_argument('--seed',type=int,default=0);p.add_argument('--output',type=Path)
    a=p.parse_args();r=probe(a.mode,a.seed)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({'seed':a.seed,'mode':a.mode,'label':r['result']['label'],
                      'error':r.get('error'),'wall_s':r.get('wall_s')},indent=2))
