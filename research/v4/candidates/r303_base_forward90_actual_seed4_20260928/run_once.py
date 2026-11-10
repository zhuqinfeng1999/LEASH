"""Root-only one new physical-version R303 seed4 naive script binding.

Original selected script/controller/evaluator remain unchanged. This file
only selects an opt-in constructor and records one fresh result.
"""
import argparse,hashlib,json,time
from pathlib import Path
from unittest.mock import patch
from research.v4.candidates.r3_03_own_horizontal_actual_scripts_20260928 import entry as original
from research.v4.candidates.r303_base_forward90_copy_20260928.scene import BaseForward90OwnThermosScene

H=Path(__file__).resolve().parent;ROOT=H.parents[3]
TRIAL='R303+90mm-base-new-physical-version-seed4-naive-once'

def make_scene(seed=0,*,camera_observations=False,render_gpu_device_id=-1):
    return BaseForward90OwnThermosScene(seed,camera_observations=camera_observations,
                                       render_gpu_device_id=render_gpu_device_id,
                                       root_base_forward_approved=True)

def hashes():
    packet=json.loads((H/'source_freeze.json').read_text())
    for name,digest in packet.items():
        p=Path(name);p=p if p.is_absolute()else ROOT/p
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('R303 new physical source drift: '+name)
    original.admission()
    return packet

def selected():
    f,src,contract=original.selected()
    if f.__globals__['R3TabletopFacade'] is not original.SelectedOwnThermosScene:
        raise ValueError('Original selected R303 Scene seam changed')
    if f.__globals__['MAX_STEPS']!=1200 or f.__globals__['MAX_WALL_S']!=90.:
        raise ValueError('Original task control/wall caps changed')
    return f,src,contract

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--seed',type=int,default=4)
    ap.add_argument('--output')
    a=ap.parse_args();packet=hashes();f,src,contract=selected()
    if a.preflight_only:
        print(json.dumps({'trial':TRIAL,'pinned_source_count':len(packet),'original_selected_source_sha256':hashlib.sha256(src.encode()).hexdigest(),'selected_original_scene':f.__globals__['R3TabletopFacade'].__name__,'new_factory':BaseForward90OwnThermosScene.__name__,'branch':'naive','seed':4,'original_caps':[1200,90.],'Scene_constructors':0,'task_controls':0,'GPU':False,'model_or_method_result':False}));return
    if a.seed!=4 or not a.output:raise ValueError('Only the original failed seed4 naive one-time trial is permitted')
    out=Path(a.output).resolve()
    if out.exists():raise ValueError('Fresh output required: never overwrite failure or retry')
    out.mkdir(parents=True)
    (out/'selected_original_script.py.txt').write_text(src)
    report={'trial':TRIAL,'seed':4,'case':'naive','new_physical_layout':BaseForward90OwnThermosScene.layout_version,'original_script_source_sha256':hashlib.sha256(src.encode()).hexdigest(),'source_sha256':packet,'original_caps':{'controls':1200,'wall_s':90.},'models_or_methods_called':False,'method_count_added':0,'first_failure_stop':True,'result_path':str(out/'naive_seed4.json'),'error':None}
    started=time.monotonic()
    try:
        with patch.dict(f.__globals__,{'R3TabletopFacade':make_scene}):
            if f.__globals__['R3TabletopFacade'] is not make_scene:raise RuntimeError('Selected original generated main did not consume opt-in factory')
            result=f(4,'naive')
        (out/'naive_seed4.json').write_text(json.dumps(result,indent=2)+'\n')
        report['outcome']={'passed':bool(result['passed']),'label':result['evaluator']['label'],'controls':result['control_steps'],'script_error':result['error'],'true_bind':result.get('true_bind'),'held_after_lift':result.get('held_after_lift')}
    except BaseException as e:report['error']=f'{type(e).__name__}: {e}'
    finally:
        report['elapsed_s']=time.monotonic()-started
        report['source_drift']=[name for name,digest in packet.items()if hashlib.sha256((Path(name)if Path(name).is_absolute()else ROOT/name).read_bytes()).hexdigest()!=digest]
        (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report.get(k)for k in ('trial','outcome','error','source_drift','elapsed_s')}))
    if report['error'] or report.get('source_drift') or not report.get('outcome',{}).get('passed'):raise SystemExit(2)

if __name__=='__main__':main()
