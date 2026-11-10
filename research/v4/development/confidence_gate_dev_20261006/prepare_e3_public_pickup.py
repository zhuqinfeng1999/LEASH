"""Reuse the existing E301 public handle estimator and six RPent calls.

This is a preparation seam, not a registered tool or controller. Root supplies
one actual same-step public packet and the already-running SAM3 client. No
Scene, capture, physics step, model service or old Consumer is constructed.
"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from research.v4.accelerated.e3_public_pick import HandleEstimate, estimate_handle

ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT/'research/v4/runs/lite_20261003/E3_01_workpackage_pour_same_direction'


def initial_saved_public_packet(state, *, executed_sequence):
    """Read the existing RPent initial dump while the Owner is paused.

    Artifact steps count tool observations, not control steps. This narrow
    seam accepts only the fresh, pre-control dump; it never relabels a later
    observation with a guessed control sequence or calls EGL off its Owner.
    """
    if type(executed_sequence) is not int or executed_sequence != 0 or state.latest_step != 0:
        raise ValueError('E3 pickup requires the actual untouched initial public dump')
    from research.v4.bridge.observation import validate_state_step
    from research.v4.accelerated.observation import public_observation
    validate_state_step(state,0)
    record=state.get(0)
    if record.step_idx != 0:
        raise ValueError('Initial artifact identity differs')
    raw=record.state
    robot={name:raw['robot0_'+name] for name in ('eef_pos','eef_quat','gripper_qpos')}
    cameras={}
    for saved,name in (('agentview','agentview'),('wrist','robot0_eye_in_hand')):
        meta=state.load(saved+'_metadata.json',step=0)
        rgb=np.asarray(state.load(saved+'.png',step=0))
        depth=np.asarray(state.load(saved+'_depth.npz',step=0)).squeeze()
        cameras[name]=dict(rgb=rgb,depth_m=depth,
            intrinsics=np.asarray(meta['intrinsic_K']),
            extrinsics=np.asarray(meta['extrinsic_cam2world']),frame_id=0)
    return public_observation(cameras,robot,frame_id=0)


def pickup_stages(estimate):
    """The first six actual e3_lite.pickup_actions, unchanged numbers/order."""
    if not isinstance(estimate, HandleEstimate):
        raise TypeError('Use the original HandleEstimate')
    xyz = np.asarray(estimate.xyz_m, dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError('Finite original public handle estimate required')
    x,y,z = map(float,xyz)
    return [
        dict(tool='move_to',arguments=dict(xyz=[-.25,y,1.18],gripper=-1.,max_steps=180,tol=.012,step_clip=.012)),
        dict(tool='move_to',arguments=dict(xyz=[x,y,1.18],gripper=-1.,max_steps=180,tol=.010,step_clip=.012)),
        dict(tool='move_to',arguments=dict(xyz=[x,y,1.04],gripper=-1.,max_steps=180,tol=.008,step_clip=.007)),
        dict(tool='move_to',arguments=dict(xyz=[x,y,z],gripper=-1.,max_steps=140,tol=.005,step_clip=.005)),
        dict(tool='set_gripper',arguments=dict(gripper=1.,steps=26)),
        dict(tool='move_to',arguments=dict(xyz=[x,y,1.10],gripper=1.,max_steps=180,tol=.012,step_clip=.006)),
    ]


def prepare_public_pickup(packet, sam_client, *, expected_sequence):
    """One original public SAM query; no default/forced holding state."""
    if (type(expected_sequence) is not int or expected_sequence < 0 or
            packet.get('frame_id') != expected_sequence):
        raise ValueError('Public pickup and actual executed sequence differ')
    robot = packet.get('robot_state')
    if not isinstance(robot,dict):
        raise ValueError('Same-packet public proprioception missing')
    for key,size in (('eef_pos',3),('eef_quat',4),('gripper_qpos',2)):
        value = np.asarray(robot.get(key),dtype=float)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError('Public '+key+' missing')
    if not .9 < float(np.linalg.norm(robot['eef_quat'])) < 1.1:
        raise ValueError('Public body-world xyzw quaternion invalid')
    # Uses exactly the installed original estimator, including its existing
    # published-kettle surface calibration. No new point compensation.
    estimate = estimate_handle(packet,sam_client,'grey kettle handle')
    return estimate,pickup_stages(estimate)


def save_public_pick(output, estimate):
    """The existing public_pick.json record, not a new evidence format."""
    path=Path(output)/'public_pick.json'
    with path.open('x') as stream:
        json.dump(asdict(estimate),stream,indent=2,allow_nan=False)
        stream.write('\n')
    return path


def saved_example():
    """Read the closed example for preparation only; never reuse its permit."""
    raw=json.loads((EXAMPLE/'public_pick.json').read_text())
    estimate=HandleEstimate(**{**raw,'xyz_m':tuple(raw['xyz_m']),
                                'visible_span_m':tuple(raw['visible_span_m'])})
    stages=pickup_stages(estimate)
    saved=json.loads((EXAMPLE/'scripted_tools.json').read_text())
    if stages != [dict(tool=r['tool'],arguments=r['arguments']) for r in saved[:6]]:
        raise ValueError('Saved example differs from the original six pickup calls')
    return dict(source_run=str(EXAMPLE.relative_to(ROOT)),instance='E3-01',seed=0,
        source_public_estimate=raw,stages=stages,
        source_phase='scripted_public_perception_engineering',
        script_not_registered=True,natural_method_result=False,formal=False,
        new_execution_required=True,private_latch_not_used=True,
        source_custody=False,model_requests_started=0,native_starts=0,
        source_lift_return=dict(final_dist_m=saved[5]['result']['log']['result']['final_dist_m'],
            requested_tolerance_m=saved[5]['arguments']['tol'],
            final_public_eef_pos=saved[5]['result']['state']['robot0_eef_pos']),
        reuse_scope='Recompute one handle estimate from the actual new initial public packet; the archived xyz is only an example, not the new execution target.',
        actor_seam='At first iteration of current scripted.actions, after stack.begin_round and before the first primitive, Root uses the holder stack public capture and existing sam3_client to prepare stages. Submit each yielded existing tool via the unchanged Owner. Existing failed_return/withholding still controls the sequence.',
        rotation_scope='Only pickup is prepared here. rotate_pitch and move_pose pitch drive world-X action[3]; old world-Y action[4] engineering tilts are not equivalent. No tilt suffix is supplied.',
        historical_success_not_inherited=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.output.open('x') as stream:
        json.dump(saved_example(),stream,indent=2,allow_nan=False)
        stream.write('\n')
