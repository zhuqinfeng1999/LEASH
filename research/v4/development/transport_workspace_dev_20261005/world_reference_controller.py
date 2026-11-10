"""Development candidate: original OSC follows the declared world reference.

Instance-local around a common analytic translation only. Original OSC goal
parsing, gains, actuator limits and VLA delta controls outside this context
remain unchanged. No new planner or physics update is introduced.
"""
from contextlib import contextmanager
import numpy as np
from scipy.spatial.transform import Rotation

VERSION='common-translation-persistent-world-reference-dev2-20261005'


def body_to_site_rotation(model,hand,site):
    # The installed gripper may put grip_site on a fixed descendant of the
    # robot right_hand body. Use only static robot installation transforms.
    body=int(model.site_bodyid[site]);chain=[]
    while body!=hand:
        if body<=0 or int(model.body_jntnum[body])!=0:
            raise ValueError('grip_site has no fixed installation path from right_hand')
        chain.append(body);body=int(model.body_parentid[body])
    R=np.eye(3)
    for child in reversed(chain):
        q=np.asarray(model.body_quat[child],float)
        R=R @ Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    q=np.asarray(model.site_quat[site],float)
    return R @ Rotation.from_quat(q[[1,2,3,0]]).as_matrix()


def to_controller_frame(position, rotation, controller):
    if controller.input_ref_frame == 'world':
        return np.array(position), np.array(rotation)
    if controller.input_ref_frame == 'base':
        return (controller.origin_ori.T @ (np.array(position)-controller.origin_pos),
                controller.origin_ori.T @ np.array(rotation))
    raise ValueError('Unsupported existing OSC reference frame')


@contextmanager
def persistent_reference(scene, description, trace, *, primitives=None):
    robot=scene.env.env.robots[0]
    c=robot.part_controllers['right']
    if (c.input_type!='delta' or c._goal_update_mode!='achieved'
            or c.impedance_mode!='fixed' or c.interpolator_pos is not None
            or c.interpolator_ori is not None):
        raise ValueError('Candidate requires the existing uninterpolated fixed OSC')
    model=scene.backend.sim.model
    site=robot.eef_site_id['right']
    hand=model.body_name2id(robot.robot_model.eef_name['right'])
    body_to_site=body_to_site_rotation(model,hand,site)
    fixed_site_R=np.asarray(description.start.rotation) @ body_to_site
    original=c.set_goal
    own='set_goal' in c.__dict__; prior=c.__dict__.get('set_goal')
    if primitives is not None and hasattr(primitives,'_persistent_world_reference_descriptor'):
        raise ValueError('Nested analytic world-reference context')
    def set_goal(action):
        answer=original(action)  # original update and scaling, once
        if not trace:
            raise ValueError('No current common analytic reference')
        row=trace[-1]
        ref=np.asarray(row['reference_position'],float)
        expected=description.reference(int(row['control'])+1)[1]
        if not np.array_equal(ref,expected):
            raise ValueError('Reference differs from this exact prepared line')
        p,r=to_controller_frame(ref,fixed_site_R,c)
        c.goal_pos=p; c.goal_ori=r
        row['encoded_delta_target_position']=row['controller_target_position']
        row['encoded_delta_target_rotation']=row['controller_target_rotation']
        row['controller_target_position']=ref.tolist()
        row['controller_target_rotation']=[list(v) for v in description.start.rotation]
        row['actual_controller_variant']=VERSION
        return answer
    c.set_goal=set_goal
    if primitives is not None:primitives._persistent_world_reference_descriptor=description
    try:
        yield
    finally:
        if own:c.set_goal=prior
        else:del c.set_goal
        if primitives is not None:del primitives._persistent_world_reference_descriptor
