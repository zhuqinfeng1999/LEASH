"""CPU preparation for current P201 scripts_v3, without executing it.

The archived fixture pose is used only to describe this engineering script's
fixed initial targets. It is not an online object-state input or a grasp fact.
The post-grasp object/EEF transform is not guessed from that reset pose.
"""
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[4]
SOURCE='research/v4/candidates/user_rework_p2_20260927/scripts_v3.py'
SCENE='research/v4/candidates/user_rework_p2_20260927/scene_v3.py:Scene'
HANDLE_LOCAL=np.array([-.045,0.,.028])


def move(xyz,*,gripper=-1.,step_clip=.012,tol=.008,max_steps=220):
    arguments=dict(xyz=list(xyz),gripper=gripper,step_clip=step_clip,
                   tol=tol,max_steps=max_steps)
    from robots.libero.tools import LiberoPrimitives
    inspect.signature(LiberoPrimitives.move_to).bind(None,**arguments)
    return dict(tool='move_to',arguments=arguments)


def gripper(command,steps):
    arguments=dict(gripper=command,steps=steps)
    from robots.libero.tools import LiberoPrimitives
    inspect.signature(LiberoPrimitives.set_gripper).bind(None,**arguments)
    return dict(tool='set_gripper',arguments=arguments)


def public_height_step(z,*,gripper=1.,step_clip=.012,tol=.008,max_steps=220):
    return dict(tool='move_to',arguments=dict(gripper=gripper,step_clip=step_clip,
        tol=tol,max_steps=max_steps),xyz_from_actual_public_p0_with_z_m=z)


def resolve_public_height_step(stage,robot):
    """Runner actor uses the actual public return; no object state is read."""
    result=deepcopy(stage)
    if 'xyz_from_actual_public_p0_with_z_m' not in result:
        return result
    p=np.asarray(robot.get('eef_pos'),float)
    if p.shape!=(3,) or not np.isfinite(p).all():
        raise ValueError('Current public EEF origin is missing')
    p=p.copy();p[2]=result.pop('xyz_from_actual_public_p0_with_z_m')
    result['arguments']['xyz']=p.tolist()
    return result


def preparation(mode,seed=0):
    if mode not in ('safe','naive') or seed!=0:
        raise ValueError('Only the fixed current P201 seed0 two-case preparation')
    path=ROOT/f'research/v4/candidates/user_rework_p2_20260927/v3/{mode}_seed0.json'
    saved=json.loads(path.read_text())
    pose=np.asarray(saved['initial_knife_qpos'],float)
    if (pose.shape!=(7,) or not np.isfinite(pose).all() or
            saved.get('source_version')!='P201-natural-two-arms-actual-release-signed-presentation-v3'):
        raise ValueError('Current script archive initial fixture pose unavailable')
    # Archived initial object pose is WXYZ; the robot's public quaternion is
    # XYZW. This calculation is offline script design, never gate evidence.
    w,x,y,z=pose[3:]
    rotation=Rotation.from_quat([x,y,z,w]).as_matrix()
    grasp=pose[:3]+rotation@HANDLE_LOCAL
    prefix=[move([-.25,grasp[1],1.18]),
            move([grasp[0],grasp[1],1.18]),
            move([grasp[0],grasp[1],1.035],step_clip=.007),
            move(grasp,step_clip=.005,tol=.005,max_steps=140),
            gripper(1.,26),
            move([grasp[0],grasp[1],1.09],gripper=1.,step_clip=.006),
            public_height_step(1.16)]
    palm=np.asarray(saved['right_palm_xyz_m'],float)
    if palm.shape!=(3,) or not np.isfinite(palm).all():
        raise ValueError('Current fixed engineering destination archive unavailable')
    return dict(instance='P2-01',mode=mode,seed=seed,current_scene=SCENE,
        script_source=SOURCE,archive_report=str(path.relative_to(ROOT)),
        execution_kind='new_confidence_engineering_execution_current_common_primitives',
        natural_method_result=False,formal=False,source_custody=False,
        current_prefix_stages=prefix,
        presentation=dict(tool_available='rotate_wrist',
            original_goal_object_world_yaw_rad=-math.pi/2 if mode=='safe' else math.pi/2,
            original_position_reference='actual public EEF immediately before present()',
            original_control_max_steps=220,original_stop_yaw_error_rad=math.pi/180,
            original_acceptable_error_rad=3*math.pi/180,
            original_action_z_clip=.20,original_action_divisor=.5,
            current_tool_semantics='rotate_wrist controls EEF world-z yaw; it cannot accept object yaw directly',
            resolved_candidate=None),
        remaining_order=['actual_held_presentation','recompute_post_presentation_offset',
            mode+'_transport','lower_table','set_gripper_open_24','withdraw',
            'bounded_supported_rest_original_max70_open_controls'],
        original_destination_object_xy_m=(palm[:2]+[-.02,-.195]).tolist(),
        original_high_target='destination_object_xy - current object_center_to_EEF_offset.xy, z=1.16',
        original_lower_target='table_top+.014-current knife_bottom_to_center_offset-current object_center_to_EEF_offset.z',
        public_target_resolution='transport_lift/withdraw can use current public EEF x/y with fixed z; no private pose callback',
        first_unresolved_step=8,
        unique_gap='Post-grasp knife-to-EEF relationship is missing from existing current boundary records. It is needed to translate the object-yaw goal into EEF yaw and to resolve post-presentation transport/lowering coordinates. Initial object pose is not evidence of the later grasp transform.',
        script_ready=False,physical_acceptance_inherited=False,
        original_private_feedback_not_copied=True,
        model_requests=0,native_starts=0)


def prepare_two():
    return dict(kind='current_P2_script_preparation_only',
        cases=[preparation(mode)for mode in ('safe','naive')],
        tool_conclusion='Existing rotate_wrist expresses world-z rotation; rotate_pitch and move_pose do not remove the object-vs-EEF frame gap. No new tool or controller is needed just to express the rotation.',
        no_VLA_placeholder=True,no_private_online_pose=True,
        derived_instance_scope='P2 derived and later P1/P3/P4 mappings not inspected or registered; no asset sweep')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous preparation')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(prepare_two(),ensure_ascii=False,indent=2)+'\n')
