"""Offline acceptance of the two newly requested fresh-reset seeds only.

Private initialization geometry is checked here and never sent online. The
runtime issuer will rely on this fixed construction recipe, not inspect
dynamic object/attachment state to produce an occupancy answer.
"""
from contextlib import nullcontext
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from .native import ROOT, robot_job


def main(output, gpu):
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID=str(gpu),
        SAFEVLA_SOURCE_CUSTODY='0', SAFEVLA_RUN_OUTPUT=str(output))
    from research.v4.bridge.run_b0 import bind_rpent_checkout
    bind_rpent_checkout()
    from .runner import p1_configuration
    from research.v4.development.p102_contact_repair_20261005.binding import binding_scope
    from research.v4.development.transport_workspace_dev_20261005.progress import claim
    from research.v4.leash_v43.run_manifest import run_manifest, atomic_json
    from research.v4.leash_v43.initial_occupancy import VERSION
    from research.v4.tasks.p1.evaluator import _combined_aabb
    from PIL import Image
    start=time.monotonic(); rows=[]
    def expire(*_):raise TimeoutError('180-second fresh-reset acceptance limit')
    previous=signal.signal(signal.SIGALRM,expire)
    sources=('research/v4/development/confidence_gate_dev_20261006/accept_initial_seeds.py',
        'research/v4/development/confidence_gate_dev_20261006/runner.py',
        'research/v4/development/p1_single_right_arm_20261004/layout.py',
        'research/v4/development/p102_contact_repair_20261005/binding.py')
    try:
        with robot_job(),run_manifest(output,repo=ROOT,method='offline_reset_acceptance',
            instance_id='P1-02',seed=-1,gates={},planner_model=None,planner_reasoning=None,
            reviewer_model=None,reviewer_reasoning=None,source_files=sources,
            run_type='confidence_reset_development',budgets={'wall_s':180,'output_bytes':100*1024**2}):
            signal.setitimer(signal.ITIMER_REAL,180)
            configured=p1_configuration('P1-02',known_empty=False)
            for seed in (1,2):
                claim('native_engineering_trajectories')
                with binding_scope():
                    scene=configured.Scene(seed,camera_observations=True,render_gpu_device_id=gpu)
                try:
                    sim=scene.backend.sim
                    geoms=[sim.model.geom_id2name(i) for i in range(sim.model.ngeom)
                           if (sim.model.geom_id2name(i) or '').startswith('gripper0_') and sim.model.geom_contype[i]]
                    hand=_combined_aabb(sim,geoms)
                    gaps={}
                    for name,obj in scene.backend.objects_dict.items():
                        box=_combined_aabb(sim,obj.contact_geoms)
                        gaps[name]=float(np.linalg.norm(np.maximum(0.,np.maximum(hand[0]-box[1],box[0]-hand[1]))))
                    active=[bool(scene.grasp_assist.data.eq_active[e[3]]) for e in scene.grasp_assist._entries]
                    row=dict(instance='P1-02',seed=seed,sequence=scene.sequence,time_s=float(sim.data.time),
                        gap_lower_bounds_m=gaps,assist_connections_active=active,
                        assist_held_index=scene.grasp_assist._held_index,
                        robot_max_speed=float(np.max(np.abs(sim.data.qvel[:9]))),
                        no_prior_owner_or_queue=True,fresh_constructor=True)
                    row['accepted']=(row['sequence']==0 and row['time_s']==0 and min(gaps.values())>0
                        and not any(active) and row['assist_held_index'] is None and row['robot_max_speed']==0)
                    packet=scene.capture_agent()
                    for view,camera in packet['cameras'].items():
                        Image.fromarray(camera['rgb']).save(output/f'seed{seed}_{view}.png')
                    rows.append(row)
                    atomic_json(output/'results.json',dict(protocol=VERSION,rows=rows,
                        physical_controls=0,models=0,offline_only=True,online_dynamic_truth=False,
                        original_counterexamples='research/v4/runs/verified_empty_reset_20261005/initialization_acceptance/results.json'))
                    if not row['accepted']:raise RuntimeError('Fresh reset did not satisfy existing recipe: '+str(seed))
                finally:scene.close()
                if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>100*1024**2:
                    raise RuntimeError('100-MiB fresh-reset acceptance output limit')
            atomic_json(output/'accounting.json',dict(wall_s=time.monotonic()-start,
                bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
                files=sum(1 for p in output.rglob('*') if p.is_file()),
                utc=datetime.now(timezone.utc).isoformat()))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    print(json.dumps({'accepted':[r['seed'] for r in rows if r['accepted']],'wall_s':time.monotonic()-start}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=1)
    args=parser.parse_args();main(args.output.resolve(),args.gpu)
