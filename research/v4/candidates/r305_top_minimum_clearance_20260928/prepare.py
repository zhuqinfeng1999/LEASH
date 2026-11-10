"""One causal shorter approach on the original copied and actual top helpers."""
import hashlib
import json
from pathlib import Path
from types import FunctionType

from research.v4.candidates.r305_standard_top_copy_20260928 import check as parent
from research.v4.candidates.r305_standard_top_copy_20260928 import actual_expert,actual_entry

H=Path(__file__).resolve().parent
ROOT=H.parents[3]
VERSION='R305-standard-top-derived-minimum-source-clearance-OSC-v2'

def clearance(m,d,center,R,target,result):
    home,above=parent.clearance_waypoint(m,d,center,R,target,result)
    proof=result['source_clearance_waypoint']
    above[2]=proof['minimum_whole_shape_clearance_EEF_z_m']
    proof.update(EEF=above.tolist(),reused_actual_home_height=False,
        one_derived_minimum_height=True,no_new_height_grid=True,
        actual_clearance_beyond_12mm_residual_m=float(above[2]+proof['lowest_hand_wrist_relative_z_m']-proof['highest_source_and_board_m']-.012),
        original_high_lateral_waypoint_shortened_m=float(home[2]-above[2]))
    return home,above

def copied():
    fn,source=parent.selected()
    log=fn.__globals__['_ik']
    local_log=FunctionType(log.__code__,dict(log.__globals__,H=H,IK_LOG=[]),log.__name__,log.__defaults__,log.__closure__)
    return FunctionType(fn.__code__,dict(fn.__globals__,H=H,clearance_waypoint=clearance,_ik=local_log),fn.__name__,fn.__defaults__,fn.__closure__),source

def naive():
    fn=actual_expert.run_naive
    return FunctionType(fn.__code__,dict(fn.__globals__,clearance_waypoint=clearance),fn.__name__,fn.__defaults__,fn.__closure__)

def selected():
    fn,source=actual_entry.selected()
    if source.count(actual_entry.VERSION)!=1:raise RuntimeError('Original own case version seam changed')
    source=source.replace(actual_entry.VERSION,VERSION)
    ns=dict(fn.__globals__,run_naive=naive())
    exec(compile(source,str(H/'prepare.py')+'::same-original-case','exec'),ns)
    return ns['run_case'],source

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def admission():
    pins=json.loads((H/'source_freeze.json').read_text())
    for n,s in pins.items():
        p=Path(n)if n.startswith('/')else ROOT/n
        if sha(p)!=s:raise RuntimeError('Minimum-clearance source drift '+n)
    actual_entry.admission()
    r=json.loads((H/'report.json').read_text())
    if not r['passed'] or r.get('error') or r['copied_geometric_lift']['payload_Z_delta_m']<.020:raise RuntimeError('Shorter copied route unavailable')
    if any(c['hard_stop']for row in r['route']for c in row['contacts']):raise RuntimeError('Copied short route hard contact')
    return pins

def preflight():
    pins=admission();fn,source=selected()
    if fn.__globals__['MAX_STEPS']!=1200 or fn.__globals__['MAX_WALL_S']!=90.:raise RuntimeError('Original budget changed')
    if source!=(H/'selected_actual_case_source.txt').read_text():raise RuntimeError('Selected case source changed')
    if actual_expert.SELECTED_NAIVE_SOURCE!=(H/'selected_actual_naive_source.txt').read_text():raise RuntimeError('Actual driver source changed')
    return {'passed':True,'pins':len(pins),'only_minimum_clearance_waypoint':True,'ordinary_driver_caps_risk_assist_unchanged':True,'Scene_controls_GPU_models':0}

def main():
    fn=FunctionType(actual_entry.main.__code__,dict(actual_entry.main.__globals__,H=H,VERSION=VERSION,
        preflight=preflight,admission=admission,selected=selected,SELECTED_NAIVE_SOURCE=actual_expert.SELECTED_NAIVE_SOURCE))
    return fn()

if __name__=='__main__':main()
