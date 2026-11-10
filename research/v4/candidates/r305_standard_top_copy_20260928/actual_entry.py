"""Root-only prospective one-seed trio; no automatic seeds/adoption."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

from research.v4.candidates.r305_robot_facing_lug_y80_copy_20260928 import actual_entry as parent
from .actual_expert import initial,journal,run_naive,SELECTED_NAIVE_SOURCE

H=Path(__file__).resolve().parent
ROOT=H.parents[3]
VERSION='R305-supportedY80-original-ear-standard-top-actual-OSC-v1'


def selected():
    original,source=parent.selected()
    old=parent.VERSION
    if source.count(old)!=1:raise RuntimeError('Own actual case version seam differs')
    source=source.replace(old,VERSION)
    namespace=dict(original.__globals__,run_naive=run_naive,initial=initial,journal=journal)
    exec(compile(source,str(H/'actual_entry.py')+'::same-own-case-top-expert','exec'),namespace)
    fn=namespace['run_case']
    for name in ('R3TabletopFacade','Evaluator'):
        if fn.__globals__[name]is not original.__globals__[name]:
            raise RuntimeError('Original actual own Scene/evaluator replaced '+name)
    return fn,source


def admission():
    pins=json.loads((H/'actual_source_freeze.json').read_text())
    for name,value in pins.items():
        path=Path(name)if name.startswith('/')else ROOT/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=value:raise RuntimeError('Source drift '+name)
    copied=json.loads((H/'report.json').read_text())
    if not copied['passed']or not copied['route_closure']['eligible']or copied['copied_geometric_lift']['payload_Z_delta_m']<.020:
        raise RuntimeError('Fixed top copied prerequisite unavailable')
    if any(c['hard_stop']for row in copied['route']for c in row['contacts']):
        raise RuntimeError('Complete paired held copied route has a hard STOP')
    if not any(row['original_critical']for row in copied['route']):
        raise RuntimeError('Original copied critical absent')
    return pins


def preflight():
    pins=admission();fn,case_source=selected()
    if fn.__globals__['MAX_STEPS']!=1200 or fn.__globals__['MAX_WALL_S']!=90.:
        raise RuntimeError('Original actual total controls/wall changed')
    for filename,source in [('selected_actual_case_source.txt',case_source),
                            ('selected_actual_naive_source.txt',SELECTED_NAIVE_SOURCE)]:
        if source!=(H/filename).read_text():raise RuntimeError('Frozen actual selection source differs '+filename)
    writes=[]
    for node in ast.walk(ast.parse(SELECTED_NAIVE_SOURCE)):
        if isinstance(node,(ast.Assign,ast.AugAssign)):
            for target in node.targets if isinstance(node,ast.Assign)else[node.target]:
                if any(key in ast.unparse(target)for key in ('qpos','eq_data','eq_active','ctrl[')):
                    writes.append(ast.unparse(target))
    if writes:raise RuntimeError('Unexpected actual expert state writes '+str(writes))
    return dict(passed=True,pins=len(pins),selected_factory=fn.__globals__['R3TabletopFacade'].__module__+'.'+fn.__globals__['R3TabletopFacade'].__name__,
        selected_evaluator=fn.__globals__['Evaluator'].__module__+'.'+fn.__globals__['Evaluator'].__name__,
        original_driver_and_top_material_identity=True,actual_intermediate_driver_tol_m=.008,
        copied_default_position_tol_m=.012,actual_final_material_tol_m=.001,rotation_tol_deg=1,
        requested_lift_waypoint_m=.032,original_real_lift_requirement_m=.020,
        total_declared_phase_controls_including_reset=20+180+180+160+24+180+220+220,
        original_caps=[1200,90.],common_reset_controls=20,live_state_writes=writes,
        Scene_constructors=0,actual_task_controls=0,GPU_models_services=0,actual_trio_unrun=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true')
    p.add_argument('--seed',type=int,default=2);p.add_argument('--output');a=p.parse_args()
    if a.preflight:print(json.dumps(preflight(),indent=2));return
    preflight();pins=admission();fn,source=selected()
    if not a.output:raise RuntimeError('Root-only fresh output required')
    out=Path(a.output).resolve()
    if out.exists():raise RuntimeError('No overwrite/retry of task evidence')
    out.mkdir(parents=True);(out/'selected_case_source.txt').write_text(source)
    (out/'selected_naive_source.txt').write_text(SELECTED_NAIVE_SOURCE)
    receipt=dict(seed=a.seed,instance='R3-05',script_version=VERSION,cases=[],source_sha256=pins,
        first_failure_stop=True,root_only_actual_task_scripts=True,method_count_excluded=True,
        original_global_controls_including_reset=1200,original_case_wall_s=90.,actual_trio_unrun_at_preparation=True)
    started=time.monotonic()
    try:
        for mode in ('correct','naive','capability_failure'):
            row=fn(a.seed,mode);f=out/(mode+'_seed'+str(a.seed)+'.json')
            f.write_text(json.dumps(row,indent=2)+'\n')
            receipt['cases'].append(dict(case=mode,passed=row['passed'],label=row['evaluator']['label'],
                controls=row['control_steps'],total_controls=row['total_controls_including_common_reset'],
                error=row['error'],output_SHA256=hashlib.sha256(f.read_bytes()).hexdigest()))
            print(json.dumps(receipt['cases'][-1]),flush=True)
            if not row['passed']:break
    finally:
        receipt['elapsed_s']=time.monotonic()-started
        receipt['source_drift']=[n for n,s in pins.items()if hashlib.sha256((Path(n)if n.startswith('/')else ROOT/n).read_bytes()).hexdigest()!=s]
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
