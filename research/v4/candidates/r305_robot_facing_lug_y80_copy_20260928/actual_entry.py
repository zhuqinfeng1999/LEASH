"""Root-only single fresh trio; child preparation does not execute task cases."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

from research.v4.candidates.r305_tall_microwave_inside12_20260928 import entry as parent
from .scene import RobotFacingLugY80Scene
from .actual_expert import initial,journal,run_naive,SELECTED_NAIVE_SOURCE

H=Path(__file__).resolve().parent
ROOT=H.parents[3]
VERSION='R305-robot-facing-ear-Y80-side-actual-OSC-v1'


def selected():
    original,src=parent.selected()
    old='R305-positive-lug-tall350-inside12-actual-OSC-v1'
    if src.count(old)!=1:raise RuntimeError('Original own case metadata seam differs')
    src=src.replace(old,VERSION)
    ns=dict(original.__globals__,R3TabletopFacade=RobotFacingLugY80Scene,
            run_naive=run_naive,initial=initial,journal=journal)
    exec(compile(src,__file__+'::original-own-case-with-real-side-expert','exec'),ns)
    f=ns['run_case']
    if f.__globals__['Evaluator']is not parent.original.Evaluator or f.__globals__['R3TabletopFacade']is not RobotFacingLugY80Scene:
        raise RuntimeError('Actual own physical factory/evaluator not consumed')
    return f,src


def admission():
    pins=json.loads((H/'actual_entry_source_freeze_v2.json').read_text())
    for n,s in pins.items():
        p=Path(n)if n.startswith('/')else ROOT/n
        if hashlib.sha256(p.read_bytes()).hexdigest()!=s:raise RuntimeError('Source drift '+n)
    constructor=json.loads((H/'actual_constructor_seed1/report.json').read_text())
    if not constructor['passed']or constructor['common_reset_controls']!=20 or not all(constructor['scope_restored'].values()):
        raise RuntimeError('Own actual normal constructor/20reset proof unavailable')
    return pins


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--seed',type=int,default=1);p.add_argument('--output')
    a=p.parse_args();pins=admission();f,src=selected()
    if a.preflight:
        writes=[]
        for source in (SELECTED_NAIVE_SOURCE,Path(__file__).with_name('actual_expert.py').read_text()):
            for n in ast.walk(ast.parse(source)):
                if isinstance(n,(ast.Assign,ast.AugAssign)):
                    for t in n.targets if isinstance(n,ast.Assign)else[n.target]:
                        if any(k in ast.unparse(t)for k in ('qpos','eq_data','eq_active','ctrl[')):
                            writes.append(ast.unparse(t))
        if writes:raise RuntimeError('Unexpected actual expert state writes '+str(writes))
        print(json.dumps(dict(passed=True,pins=len(pins),selected_factory=RobotFacingLugY80Scene.__module__+'.'+RobotFacingLugY80Scene.__name__,
            selected_evaluator=f.__globals__['Evaluator'].__module__+'.'+f.__globals__['Evaluator'].__name__,
            original_driver_callable=parent.run_naive.__globals__['generic_driver_source'].__module__+'.generic_driver_source',
            original_case_critical_source=hashlib.sha256(parent.original.Evaluator.observe.__globals__['snapshot'].__code__.co_code).hexdigest(),
            generated_case_sha256=hashlib.sha256(src.encode()).hexdigest(),generated_naive_sha256=hashlib.sha256(SELECTED_NAIVE_SOURCE.encode()).hexdigest(),
            live_state_writes=writes,global_caps=[1200,90.],common_reset_included=20,
            maximum_declared_phase_controls_including_reset=20+180+160+160+24+180+220+220,
            Scene_constructors=0,task_controls=0,GPU_models_services=0,actual_trio_unrun=True),indent=2));return
    if not a.output:raise RuntimeError('Root-only fresh output required')
    out=Path(a.output).resolve()
    if out.exists():raise RuntimeError('No overwrite/retry of task evidence')
    out.mkdir(parents=True);(out/'selected_case_source.txt').write_text(src)
    (out/'selected_naive_source.txt').write_text(SELECTED_NAIVE_SOURCE)
    receipt=dict(seed=a.seed,instance='R3-05',script_version=VERSION,cases=[],source_sha256=pins,
        first_failure_stop=True,root_only_actual_task_scripts=True,method_count_excluded=True,
        original_global_controls_including_reset=1200,original_case_wall_s=90.)
    started=time.monotonic()
    try:
        for mode in ('correct','naive','capability_failure'):
            row=f(a.seed,mode);path=out/(mode+'_seed'+str(a.seed)+'.json')
            path.write_text(json.dumps(row,indent=2)+'\n')
            receipt['cases'].append(dict(case=mode,passed=row['passed'],label=row['evaluator']['label'],
                controls=row['control_steps'],total_controls=row['total_controls_including_common_reset'],
                error=row['error'],output_SHA256=hashlib.sha256(path.read_bytes()).hexdigest()))
            print(json.dumps(receipt['cases'][-1]),flush=True)
            if not row['passed']:break
    finally:
        receipt['elapsed_s']=time.monotonic()-started
        receipt['source_drift']=[n for n,s in pins.items()if hashlib.sha256((Path(n)if n.startswith('/')else ROOT/n).read_bytes()).hexdigest()!=s]
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
