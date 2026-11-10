"""Isolated causal candidate: honor the common runner's encoded step limits.

This candidate changes the common control implementation and is not installed
in production. No gains, model parameters, robot limits or G3 tolerances change.
"""
from contextlib import contextmanager
import numpy as np

from research.v4.development.transport_workspace_dev_20261005.world_reference_controller import (
    body_to_site_rotation, to_controller_frame,
)

VERSION='common-reference-encoded-step-bound-candidate-20261020'


@contextmanager
def bounded_reference(scene,description,trace,*,primitives):
    robot=scene.env.env.robots[0];c=robot.part_controllers['right']
    if (c.input_type!='delta' or c._goal_update_mode!='achieved'
            or c.impedance_mode!='fixed' or c.interpolator_pos is not None
            or c.interpolator_ori is not None):
        raise ValueError('Existing fixed uninterpolated OSC required')
    model=scene.backend.sim.model
    body=model.body_name2id(robot.robot_model.eef_name['right'])
    installation=body_to_site_rotation(model,body,robot.eef_site_id['right'])
    original=c.set_goal;own='set_goal' in c.__dict__;prior=c.__dict__.get('set_goal')
    def set_goal(action):
        result=original(action)
        row=trace[-1]
        if not np.array_equal(np.asarray(action,dtype=np.float32),np.asarray(row['action'][:6],dtype=np.float32)):
            raise ValueError('Controller payload differs from disclosed current vector')
        target=np.asarray(row['controller_target_position'])
        rotation=np.asarray(row['controller_target_rotation'])@installation
        c.goal_pos,c.goal_ori=to_controller_frame(target,rotation,c)
        row['actual_controller_variant']=VERSION
        return result
    c.set_goal=set_goal
    primitives._persistent_world_reference_descriptor=description
    try:yield
    finally:
        if own:c.set_goal=prior
        else:del c.set_goal
        del primitives._persistent_world_reference_descriptor


@contextmanager
def bounded_intent(dispatcher,description,trace):
    # The installed ordinary delta decoder describes this bounded target.
    # No full-reference override is allowed: review and send share one vector.
    yield
