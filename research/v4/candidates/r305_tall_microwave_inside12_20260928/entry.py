"""Root-only finite R305 trio; original correct/empty/evaluator branches."""
import argparse,ast,hashlib,inspect,json,time
from pathlib import Path
from research.v4.candidates.r3_05_fork_floor_v3_20260926 import scripts as original
from .scene import TallLugScene,H
from .expert import run_naive,initial,journal
ROOT=H.parents[3]

def selected():
    src=inspect.getsource(original.run_case)
    start=src.index('        else:\n            # The camera-ready start')
    end=src.index("        details['assist_events']",start)
    src=src[:start]+"        else:\n            run_naive(scene,evaluator,command,details,initial_z)\n"+src[end:]
    edits=[('        sim = scene.backend.sim','        sim = scene.backend.sim\n        initial(scene,details)'),
        ('            return evaluator.observe(scene)','            state=evaluator.observe(scene)\n            journal(scene,action,evaluator,details)\n            return state'),
        ('            if evaluator.samples >= MAX_STEPS:',
         '            if scene.sequence >= MAX_STEPS:'),
        ('        scene.close()',"        details['total_controls_including_common_reset']=scene.sequence\n        details['script_version']='R305-positive-lug-tall350-inside12-actual-OSC-v1'\n        scene.close()")]
    for old,new in edits:
        if src.count(old)!=1:raise RuntimeError('Actual own R305 runner seam changed '+old)
        src=src.replace(old,new)
    ns=dict(original.run_case.__globals__,R3TabletopFacade=TallLugScene,run_naive=run_naive,initial=initial,journal=journal)
    exec(compile(src,str(H/'entry.py')+'::original-own-case-material-route-config','exec'),ns)
    f=ns['run_case']
    if f.__globals__['Evaluator']is not original.Evaluator or f.__globals__['R3TabletopFacade']is not TallLugScene:
        raise RuntimeError('Own actual factory/evaluator not selected')
    return f,src

def admission():
    pins=json.loads((H/'source_freeze.json').read_text())
    for name,sha in pins.items():
        p=Path(name)if Path(name).is_absolute()else ROOT/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise RuntimeError('Source drift '+name)
    return pins

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--output');a=p.parse_args()
    pins=admission();f,src=selected()
    if a.preflight:
        tree=ast.parse((H/'expert.py').read_text());writes=[]
        for n in ast.walk(tree):
            if isinstance(n,(ast.Assign,ast.AugAssign)):
                for t in n.targets if isinstance(n,ast.Assign)else[n.target]:
                    if 'qpos'in ast.unparse(t)or'eq_data'in ast.unparse(t):writes.append(ast.unparse(t))
        if writes:raise RuntimeError('Unexpected private expert live writes '+str(writes))
        print(json.dumps({'pins':len(pins),'Scene_task_controls_GPU_models':0,
            'selected_factory':f.__globals__['R3TabletopFacade'].__qualname__,
            'selected_evaluator':f.__globals__['Evaluator'].__module__+'.'+f.__globals__['Evaluator'].__name__,
            'generated_sha256':hashlib.sha256(src.encode()).hexdigest(),'actual_trio_unrun':True,
            'caps':[1200,90.],'reset_controls_included':20,'live_state_writes':writes}));return
    if not a.output:raise RuntimeError('Fresh root-only output required')
    out=Path(a.output).resolve()
    if out.exists():raise RuntimeError('No overwrite or retry')
    out.mkdir(parents=True);(out/'selected_case_source.txt').write_text(src)
    receipt={'cases':[],'source_sha256':pins,'method_count_excluded':True,'first_failure_stop':True,
             'root_only_actual_scripts':True,'instance':'R3-05'};started=time.monotonic()
    try:
        for mode in ('correct','naive','capability_failure'):
            row=f(0,mode);(out/(mode+'_seed0.json')).write_text(json.dumps(row,indent=2)+'\n')
            receipt['cases'].append({'case':mode,'passed':row['passed'],'label':row['evaluator']['label'],
                'controls':row['control_steps'],'total_controls':row['total_controls_including_common_reset'],'error':row['error']})
            print(json.dumps(receipt['cases'][-1]),flush=True)
            if not row['passed']:break
    finally:
        receipt['elapsed_s']=time.monotonic()-started;receipt['source_drift']=[name for name,sha in pins.items()
            if hashlib.sha256((Path(name)if Path(name).is_absolute()else ROOT/name).read_bytes()).hexdigest()!=sha]
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
