"""Bounded zero-action three-view scene preflight for I1/I2."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from .core import InjectionScene
from .evaluator import InjectionEvaluator


def run(task_id, output, seed=0, size=256):
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    scene=None
    report={'task':task_id,'case':'initial_zero_action','seed':seed,
            'pid':os.getpid(),'model_calls':0,'section7_accepted':False}
    try:
        scene=InjectionScene(task_id,seed=seed,size=size)
        evaluator=InjectionEvaluator(task_id)
        report['initial_truth']=evaluator.observe(scene)
        packet=scene.observation()
        for name in scene.public_views:
            Image.fromarray(packet['cameras'][name]['rgb']).save(output/f'{name}.png')
            np.save(output/f'{name}-depth-m.npy',packet['cameras'][name]['depth_m'])
        Image.fromarray(scene.spectator_rgb()).save(output/'spectator.png')
        report['gpu_process_snapshot']=subprocess.run(
            ['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory',
             '--format=csv,noheader'],capture_output=True,text=True,timeout=5).stdout.strip()
        report['success']=not report['initial_truth']['ketchup_in_basket'] and not report['initial_truth']['target_held']
    except BaseException as error:
        import traceback
        report.update(success=False,error=str(error),traceback=traceback.format_exc())
    finally:
        if scene is not None:scene.close()
        report['elapsed_s']=time.monotonic()-start
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['success']:
        raise RuntimeError(report.get('error','I-task initial predicate failed'))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=('I1','I2'),default='I1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    print(json.dumps(run(args.task,args.output,args.seed)),flush=True)

if __name__=='__main__':main()
