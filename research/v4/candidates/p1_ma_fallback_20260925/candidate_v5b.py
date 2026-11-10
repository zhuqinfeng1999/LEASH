"""P1 v5b: v4 visible left arm, right arm extended farther out."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from unittest.mock import patch
from . import candidate_v5 as v5
class P1V5BFacade(v5.P1V5Facade):
    e1_left_rest_xyz_m = (.20,-.38,.938)
P1V5BEvaluator=v5.P1V5Evaluator
def probe(mode,seed):
    with patch.object(v5,'P1V5Facade',P1V5BFacade):
        row=v5.probe(mode,seed)
    row.update(kind='p1_v5b_natural_clearance_cpu_candidate_not_native_or_method',scene_version='p1_v5b_right_y_plus080_left_rest_v4')
    return row
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('safe','naive','failure'));p.add_argument('--seed',type=int,default=0);p.add_argument('--output',type=Path);a=p.parse_args();r=probe(a.mode,a.seed)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(a.mode,a.seed,r['result']['label'],r.get('error'))
