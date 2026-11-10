"""One bounded, model-free native diagnostic of an archived high-G3 motion.

The archived prefix is engineering setup in a freshly reset simulator. It is
never resumed as an old method episode. After setup, a new public descriptor
executes through the actual commit dispatcher using an explicitly authored
PASS/BLOCK test callback. This tests mechanics/counting, not model safety.
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[4]
RESULT = ROOT / 'research/v4/results/g3_event_count_repair_20261020'


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def main(args):
    from research.v4.development.confidence_gate_dev_20261006.formal_configurations import select
    from research.v4.development.confidence_gate_dev_20261006.native import configure_formal_private_outcome
    from research.v4.development.confidence_gate_dev_20261006.placement_outcome_dev4 import configure
    from research.v4.development.common_translation_dev_20261005.common_motion import (
        MotionDescription, PublicPose, OSCMapping, execute_motion,
    )
    from research.v4.development.transport_workspace_dev_20261005.world_reference_controller import persistent_reference
    from research.v4.development.transport_workspace_dev_20261005.controller_intent import world_reference_intent
    from research.v4.development.g3_transition_dev_20261019.scope import (
        TransitionRegistry, TransitionDispatcher, prepared_motion_scope,
    )
    from research.v4.leash_v43.adapter import OSCControlSemantics
    from research.v4.leash_v43.registry import GripperCalibration, HoldingEstimator, Proprio
    from research.v4.leash_v43.dispatch import CommitDispatcher, V43Intercepted
    from .estimator import CompletedOpenHoldingEstimator
    from .registry import CountBoundTransitionRegistry

    plan = read(ROOT/'research/v4/results/formal_g3_transition_20261020/plan.json')
    assert plan['authorization']['state'] == 'PAUSED_USER_G3_REPAIR_REQUIRED'
    attempt = next(a for a in plan['episodes'] if a['experiment_number'] == args.experiment)
    old = ROOT/attempt['output']
    traces = read(old/'control_trace.json')
    selected = next(t for t in traces if t['description']['start']['sequence'] == args.start_sequence)
    cutoff = args.start_sequence
    trace_by_sequence = {}
    for motion in traces:
        for row in motion['trace']:
            if row.get('step_returned'):
                trace_by_sequence[row['public_before']['sequence']] = (motion['description'], row)
    controls = [x for x in read(old/'dispatch.json') if x['kind'] == 'dispatched' and x['executed_sequence'] <= cutoff]
    assert [x['executed_sequence'] for x in controls] == list(range(1, cutoff+1))
    if args.output.exists():
        raise FileExistsError('Never overwrite a diagnostic: '+str(args.output))
    args.output.mkdir(parents=True)
    report = dict(kind='isolated_native_G3_count_mechanics_demo', experiment=args.experiment,
                  instance=attempt['instance'], seed=attempt['seed'], setup_controls=cutoff,
                  model_calls=0, natural_method_result=False, test_review=args.decision,
                  estimator=args.estimator, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  controller_candidate=args.controller,
                  registry_candidate=args.registry,
                  reference_scale=args.reference_scale,
                  block_at_sequence=args.block_at_sequence,
                  gpu=args.gpu, pid=os.getpid(), old_raw_sha256={n:hashlib.sha256((old/n).read_bytes()).hexdigest() for n in ['control_trace.json','dispatch.json']})
    started=time.monotonic()
    scene=None
    video_writer=None
    def timeout(*_):
        raise TimeoutError('180-second isolated native demo wall limit')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 180)
    with ExitStack() as stack:
        for name in ['robot.lock', f'robot_gpu_{args.gpu}.lock']:
            f=stack.enter_context((ROOT/'research/v4/development/transport_workspace_dev_20261005'/name).open('a'))
            fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            configured=select(attempt['instance'])
            if attempt['instance'].startswith(('P3-','I')):
                configured=configure(configured)
            configured=configure_formal_private_outcome(configured)
            os.environ['SAFEVLA_RUN_OUTPUT']=str(args.output)
            scene=configured.Scene(seed=attempt['seed'],camera_observations=True,render_gpu_device_id=args.gpu)
            semantics=OSCControlSemantics.from_scene(scene)
            e=(CompletedOpenHoldingEstimator if args.estimator=='repaired' else HoldingEstimator)(
                GripperCalibration('Existing Panda calibration',0.,.003,.075,3,.001,False),
                capture_grasp_pose_on_close=True)
            def public(sequence=None):
                raw=scene.raw
                return Proprio(float(np.abs(np.asarray(raw['robot0_gripper_qpos'])).sum()),
                               np.asarray(raw['robot0_eef_pos'],float).copy(),
                               Rotation.from_quat(raw['robot0_eef_quat']).as_matrix(),
                               scene.sequence if sequence is None else sequence)
            def descriptor(saved):
                saved=dict(saved)
                saved['start']=PublicPose(**saved['start'])
                saved['mapping']=OSCMapping(**saved['mapping'])
                return MotionDescription(**saved)
            initial_public=public()
            prefix_public=[dict(sequence=initial_public.executed_sequence,aperture_m=initial_public.aperture_m,
                                executed_gripper=None)]
            prefix_errors=[]
            for record in controls:
                seq=scene.sequence
                assert seq+1==record['executed_sequence']
                payload=np.asarray(record['payload'],np.float32)
                before=public()
                prepared=trace_by_sequence.get(seq)
                if prepared:
                    saved,row=prepared
                    prefix_errors.append(float(np.linalg.norm(before.eef_xyz-np.asarray(row['public_before']['position']))))
                    with persistent_reference(scene,descriptor(saved),[dict(row)]):
                        scene.step(payload)
                else:
                    scene.step(payload)
                command=semantics.decode(payload,before)
                e.record_command(command.gripper,before)
                after=public()
                e.observe(after)
                prefix_public.append(dict(sequence=after.executed_sequence,aperture_m=after.aperture_m,
                                          executed_gripper=command.gripper))
            actual=public()
            saved=selected['description']
            start_error=float(np.linalg.norm(actual.eef_xyz-np.asarray(saved['start']['position'])))
            rotation_error=float(Rotation.from_matrix(actual.eef_rotation@np.asarray(saved['start']['rotation']).T).magnitude())
            report['prefix_reproduction']=dict(max_saved_position_error_m=max(prefix_errors,default=0),
                selected_start_position_error_m=start_error,selected_start_rotation_error_rad=rotation_error,
                actual_holding_estimate=e.kind.value,grasp_rotation=None if e.grasp_rotation is None else e.grasp_rotation.tolist())
            report['setup_public_trace']=prefix_public
            if start_error>.002 or rotation_error>.02:
                raise RuntimeError('Archived prefix does not reproduce selected public start;do not label this an exact saved-case demo')
            description=descriptor({**saved,'start':dict(position=actual.eef_xyz.tolist(),rotation=actual.eef_rotation.tolist(),
                sequence=actual.executed_sequence,time_s=float(scene.backend.sim.data.time))})
            if not 0 < args.reference_scale <= 1:
                raise ValueError('Engineering reference fraction scale must be within(0,1]')
            description=replace(description,step_clip=description.step_clip*args.reference_scale)
            from PIL import Image
            Image.fromarray(np.asarray(scene.raw['agentview_image'])[::-1]).save(args.output/'start_agentview.png')
            if args.video:
                from imageio.v2 import get_writer
                video_writer=get_writer(str(args.output/'native_agentview.mp4'),fps=20,
                    codec='libx264',macro_block_size=None,ffmpeg_params=['-crf','18','-pix_fmt','yuv420p'])
                video_writer.append_data(np.asarray(scene.raw['agentview_image'])[::-1])
                report['video']=dict(path=str(args.output/'native_agentview.mp4'),
                    view='native public agentview',resolution=[640,480],fps=20,
                    engineering_authored_decisions=True,natural_model_result=False)
            registry=(CountBoundTransitionRegistry if args.registry=='repaired' else TransitionRegistry)(e,position_tolerance_m=.008,direction_tolerance_rad=.06,
                                        require_grasp_assumption=False,semantic_unknown_pour=True)
            trace=[];reviews=[];public_states=[]
            offline_dynamics=[]
            def read_proprio(sequence):
                p=public(sequence)
                public_states.append(dict(sequence=sequence,aperture_m=p.aperture_m,position=p.eef_xyz.tolist(),
                    rotation=p.eef_rotation.tolist(),holding=e.kind.value,generation=e.generation,
                    grasp_rotation=None if e.grasp_rotation is None else e.grasp_rotation.tolist()))
                return p
            def review(request):
                decision=('BLOCK' if dispatcher.sequence==args.block_at_sequence else args.decision)
                reviews.append(dict(sequence=dispatcher.sequence,events=list(request['events']),
                                    decision=decision))
                return dict(decision=decision,reason='Authored engineering count test;not a model judgment',
                            events=request['events'],evidence_version='engineering-fixture')
            def send_control(payload):
                answer=scene.step(payload)
                if video_writer is not None:
                    video_writer.append_data(np.asarray(scene.raw['agentview_image'])[::-1])
                return answer
            dispatcher=CommitDispatcher(registry=registry,decode=semantics.decode,read_proprio=read_proprio,
                send=send_control,capture=lambda:dict(frame_id=dispatcher.sequence),review=review,
                project=lambda **k:dict(events=k['events'],evidence_version='engineering-fixture'),
                method='Ours',enabled=True)
            dispatcher.sequence=cutoff
            class Primitive:
                def __init__(self):
                    self.env=SimpleNamespace(terminated=False,truncated=False)
                def _step_env(self,payload):
                    return dispatcher.step(payload)
            primitive=Primitive()
            def reader():
                raw=scene.raw
                return dict(robot0_eef_pos=np.asarray(raw['robot0_eef_pos']).copy(),
                            robot0_eef_quat=np.asarray(raw['robot0_eef_quat']).copy(),
                            sequence=scene.sequence,time_s=float(scene.backend.sim.data.time))
            from .offline_dynamics import collect_dynamics
            from .bounded_reference import bounded_reference, bounded_intent
            selected_controller=bounded_reference if args.controller=='bounded' else persistent_reference
            selected_intent=bounded_intent if args.controller=='bounded' else world_reference_intent
            with ExitStack() as diagnostic:
                if args.dynamics:
                    diagnostic.enter_context(collect_dynamics(scene,offline_dynamics))
                diagnostic.enter_context(dispatcher.primitive(name='native_saved_motion_demo',completion_controls=description.max_steps))
                with selected_controller(scene,description,trace,primitives=primitive),prepared_motion_scope(
                        selected_intent,dispatcher,description,trace):
                    try:
                        report['response']=execute_motion(primitive,description,trace=trace,public_reader=reader)
                        report['status']='completed'
                    except V43Intercepted:
                        expected=(args.decision=='BLOCK' or dispatcher.sequence==args.block_at_sequence)
                        if not expected or dispatcher.last_outcome.get('system_failure'):
                            raise
                        report.update(status='intercepted_expected',response=dispatcher.last_outcome)
            Image.fromarray(np.asarray(scene.raw['agentview_image'])[::-1]).save(args.output/'end_agentview.png')
            report.update(G3_calls=len(reviews),reviews=reviews,
                          public_states=public_states,description=description.as_dict(),trace=trace,
                          dispatch=dispatcher.logs,windows=registry.window_logs)
            report['offline_dynamics']=offline_dynamics
            model=scene.backend.sim.model;data=scene.backend.sim.data
            report['offline_contact_diagnostic']=[dict(geom1=model.geom_id2name(int(data.contact[i].geom1)),
                geom2=model.geom_id2name(int(data.contact[i].geom2)),distance=float(data.contact[i].dist)) for i in range(data.ncon)]
        except BaseException as exc:
            report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            if 'dispatcher' in locals():
                report.update(G3_calls=len(reviews),reviews=reviews,public_states=public_states,
                              description=description.as_dict(),trace=trace,
                              dispatch=dispatcher.logs,windows=registry.window_logs)
            raise
        finally:
            report.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.monotonic()-started)
            save(args.output/'result.json',report)
            if video_writer is not None:video_writer.close()
            if scene is not None:
                scene.close()
            signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps({k:v for k,v in report.items() if k in ['status','experiment','elapsed_s','G3_calls','prefix_reproduction','response']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=int,required=True)
    parser.add_argument('--start-sequence',type=int,required=True)
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--decision',choices=['PASS','BLOCK'],default='PASS')
    parser.add_argument('--estimator',choices=['frozen','repaired'],default='repaired')
    parser.add_argument('--reference-scale',type=float,default=1.)
    parser.add_argument('--block-at-sequence',type=int,default=-1)
    parser.add_argument('--dynamics',action='store_true')
    parser.add_argument('--controller',choices=['frozen','bounded'],default='frozen')
    parser.add_argument('--registry',choices=['frozen','repaired'],default='repaired')
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
