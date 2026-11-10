"""One no-model check of the existing R3 production stack shutdown."""
from pathlib import Path
import json
import os
import signal
import time

from research.v4.development.transport_workspace_dev_20261005.native import robot_job
from research.v4.development.transport_workspace_dev_20261005.progress import claim
from research.v4.leash_v43.run_manifest import run_manifest, atomic_json
from .family_cli_seam import select_configuration, make_private_evaluator, finish_private_evaluator
from .runner import create_episode_stack

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = 'research/v4/development/confidence_gate_dev_20261006/'
RID = 'R301_private_production_shutdown_check_kwarg_repair_seed0_20261008'
OUT = ROOT/'research/v4/runs/confidence_gate_dev_20261006'/RID
PLAN = ROOT/'research/v4/results/confidence_gate_dev_20261006/r301_private_shutdown_check_kwarg_plan_20261008.json'


def main():
    plan = json.loads(PLAN.read_text())
    assert plan['status'] == 'registered' and plan['models'] == 0
    assert plan['business_controls'] == 0 and plan['wall_limit_s'] == 180
    os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                      MUJOCO_EGL_DEVICE_ID='1', SAFEVLA_SOURCE_CUSTODY='0')
    def expire(*_):
        raise TimeoutError('180-second private shutdown check limit')
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(180)
    started = time.monotonic()
    saved = {}
    record = dict(pid=os.getpid(), models=0, business_controls=0)
    cfg = select_configuration('R3-01')
    def on_scene(scene):
        saved['scene'] = scene
        saved['evaluator'] = make_private_evaluator(cfg, scene)
        saved['evaluator'].observe(scene)
        record['initial_sequence'] = scene.sequence
    def on_execution_logs(snapshot):
        scene = saved['scene']
        record['callback_sim_alive'] = scene.backend.sim is not None
        record['capture'] = saved['evaluator'].capture_final_witness(scene)
        record['final_sequence'] = scene.sequence
        record['dispatch_count'] = len(snapshot['dispatch_logs'])
    with robot_job(), run_manifest(OUT, repo=ROOT, method='private_shutdown_check',
            instance_id='R3-01', seed=0, gates={'G1':False,'G2':False,'G3':False},
            planner_model=None, planner_reasoning=None,
            reviewer_model=None, reviewer_reasoning=None,
            source_files=[PACKAGE+name for name in (
                'r3_private_shutdown_check.py', 'r3_attachment_observation.py',
                'runner.py', 'family_cli_seam.py', 'r234_configuration.py')],
            run_type='private_production_shutdown_check',
            budgets={'wall_s':180,'output_bytes':100*1024**2,'business_controls':0}) as manifest:
        claim('native_engineering_trajectories')
        try:
            # The same create/reset, toolkit, Owner and finally as natural runs.
            # No planner/reviewer/service is constructed or tool dispatched.
            with create_episode_stack(configured=cfg, arm='B0', output=OUT, seed=0, device=1,
                    vla_client=None, sam3_client=None,
                    on_scene=on_scene, on_execution_logs=on_execution_logs):
                pass
            scene = saved['scene']
            record['backend_freed'] = scene.backend.sim is None
            record['private_result'] = finish_private_evaluator(cfg, saved['evaluator'],
                {'status':'refused'}, final_scene=scene)
            assert record['initial_sequence'] == record['final_sequence'] == 20
            assert record['dispatch_count'] == 0 and record['callback_sim_alive']
            assert record['backend_freed'] and record['capture']['status'] == 'captured'
            assert record['private_result']['ever_attached'] is False
            assert record['private_result']['dangerous_item_assist_observation'] == 'complete'
            record['status'] = 'completed'
        except BaseException as exc:
            record.update(status='failed', error=type(exc).__name__+': '+str(exc))
        finally:
            record['wall_s'] = time.monotonic()-started
            atomic_json(OUT/'diagnostic.json', record,
                        max_output_bytes=100*1024**2, budget_directory=OUT)
    signal.alarm(0)
    files = [p for p in OUT.rglob('*') if p.is_file()]
    receipt = dict(wall_s=time.monotonic()-started,
        bytes=sum(p.stat().st_size for p in files), files=len(files),
        manifest_valid=manifest['valid'], models=0, business_controls=0)
    print(json.dumps({'diagnostic':record,'receipt':receipt}), flush=True)
    return 0 if record['status']=='completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
