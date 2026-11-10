"""One Root-run no-model measurement of the saved analytic E3 prefix.

The prefix is engineering input only. No hidden geometry selects a target,
and no result enters planner/reviewer evidence or overwrites an old episode.
"""
from pathlib import Path
import json
import os
import signal
import time

from research.v4.development.transport_workspace_dev_20261005.native import robot_job
from research.v4.development.transport_workspace_dev_20261005.progress import claim
from research.v4.leash_v43.run_manifest import run_manifest, atomic_json
from research.v4.bridge.formal_recording import formal_video_scope
from .family_cli_seam import select_configuration
from .runner import create_episode_stack
from .e3_physical_event import REVISION

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = 'research/v4/development/confidence_gate_dev_20261006/'
PLAN = ROOT/'research/v4/results/confidence_gate_dev_20261006/e301_private_physical_lift_prefix_plan_20261008.json'


class MeasurementStop(Exception):
    """A predeclared post-execution measurement endpoint, not a gate."""


def main():
    plan = json.loads(PLAN.read_text())
    assert plan['status'] == 'registered' and plan['models'] == 0
    assert plan['wall_s'] == 180 and plan['output_bytes'] == 100*1024**2
    output = ROOT/'research/v4/runs/confidence_gate_dev_20261006'/plan['run_id']
    if output.exists():
        raise FileExistsError(output)
    os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                      MUJOCO_EGL_DEVICE_ID='1', SAFEVLA_SOURCE_CUSTODY='0')
    from research.v4.bridge.run_b0 import bind_rpent_checkout
    bind_rpent_checkout()
    cfg = select_configuration('E3-01')
    assert cfg.scene_event_revision == REVISION
    started = time.monotonic()
    record = dict(version=REVISION, run_id=plan['run_id'], models=0,
                  engineering_only=True, formal=False, pid=os.getpid(),
                  source_run=plan['source_run'], stages=[], stop=None)
    saved = {}
    trace = None

    def expire(*_):
        raise TimeoutError('180-second E3 private prefix check limit')

    def limits():
        if time.monotonic()-started > plan['wall_s']:
            expire()
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > plan['output_bytes']:
            raise RuntimeError('100-MiB E3 private prefix output limit')

    def observe(scene):
        row = saved['evaluator'].observe(scene)
        trace.write(json.dumps(dict(sequence=scene.sequence, **row), allow_nan=False)+'\n')
        trace.flush()
        record['physical_controls'] = scene.sequence
        limits()
        if scene.sequence > plan['max_controls']:
            raise RuntimeError('Predeclared control bound exceeded')
        if row['harm'] or row['unsafe_commit'] or row['physical_lift_qualified_now']:
            record['stop'] = ('harm_or_critical' if row['harm'] or row['unsafe_commit']
                              else 'first_qualified_lift')
            raise MeasurementStop(record['stop'])

    def on_scene(scene):
        saved.update(scene=scene, evaluator=cfg.Evaluator())
        record['initial_sequence'] = scene.sequence
        assert scene.sequence == 0
        saved['evaluator'].observe(scene)

    def on_logs(snapshot):
        atomic_json(output/'control_trace.json', snapshot['common_control_traces'])

    old_alarm = signal.signal(signal.SIGALRM, expire)
    sources = [PACKAGE+n for n in ('e3_physical_prefix_check.py','e3_physical_event.py',
               'e2_physical_lift_trigger.py','configurations.py','runner.py','common_control.py')]
    try:
        with robot_job(), run_manifest(output, repo=ROOT,
                method='offline_e3_physical_prefix', instance_id='E3-01', seed=0,
                gates={'G1':False,'G2':False,'G3':False}, planner_model=None,
                planner_reasoning=None, reviewer_model=None, reviewer_reasoning=None,
                source_files=sources, run_type='e3_private_physical_prefix_development',
                budgets={'wall_s':180,'output_bytes':plan['output_bytes']}) as manifest:
            claim('native_engineering_trajectories')
            signal.setitimer(signal.ITIMER_REAL, max(.001,180-(time.monotonic()-started)))
            trace = (output/'evaluator-only.jsonl').open('w')
            try:
                with formal_video_scope(output, enabled=True), create_episode_stack(
                        configured=cfg, arm='B0', output=output, seed=0, device=1,
                        vla_client=None, sam3_client=None, on_sim_step=observe,
                        on_scene=on_scene, on_execution_logs=on_logs) as stack:
                    primitives = stack.base.original._primitives
                    for command in plan['prefix']:
                        stage = dict(**command, sequence_before=stack.scene.sequence)
                        record['stages'].append(stage)
                        try:
                            getattr(primitives, command['tool'])(**command['arguments'])
                            stage['returned_normally'] = True
                        except MeasurementStop:
                            # The actual completed final step precedes this stop.
                            # Capture that state once without another simulation step.
                            stack.adapter._policy_obs()
                            stage['measurement_endpoint'] = True
                            break
                        finally:
                            stage['sequence_after'] = stack.scene.sequence
                        limits()
                    record['private_result'] = saved['evaluator'].finish(None)
                    record['scene_trigger_time_s'] = stack.scene.trigger_time_s
                    record['status'] = 'completed'
                    record['stop'] = record['stop'] or 'saved_prefix_end'
            except BaseException as exc:
                record.update(status='failed', error=type(exc).__name__+': '+str(exc))
            finally:
                if trace is not None:
                    trace.close()
                record['wall_s'] = time.monotonic()-started
                atomic_json(output/'diagnostic.json', record,
                            max_output_bytes=plan['output_bytes'], budget_directory=output)
        files = [p for p in output.rglob('*') if p.is_file()]
        receipt = dict(wall_s=time.monotonic()-started,bytes=sum(p.stat().st_size for p in files),
                       files=len(files),manifest_valid=manifest['valid'],models=0)
        plan.update(status=record['status'],receipt=receipt,stop=record['stop'],output=str(output.relative_to(ROOT)))
        atomic_json(PLAN,plan)
        print(json.dumps(dict(diagnostic=record,receipt=receipt)),flush=True)
        return 0 if record['status']=='completed' else 2
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_alarm)


if __name__ == '__main__':
    raise SystemExit(main())
