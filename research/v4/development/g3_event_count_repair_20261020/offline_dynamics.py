"""Read-only controller diagnostics; never passed to a gate or controller."""
from contextlib import contextmanager
import numpy as np


@contextmanager
def collect_dynamics(scene, records):
    controller=scene.env.env.robots[0].part_controllers['right']
    original=controller.run_controller
    own='run_controller' in controller.__dict__
    prior=controller.__dict__.get('run_controller')
    by_sequence={}

    def run():
        torque=original()
        c=controller;sim=scene.backend.sim
        lo,hi=c.actuator_limits
        torque=np.asarray(torque)
        joint_range=np.asarray(sim.model.jnt_range)[c.joint_index]
        margins=np.minimum(c.joint_pos-joint_range[:,0],joint_range[:,1]-c.joint_pos)
        singular=float(np.linalg.svd(c.J_full,compute_uv=False)[-1])
        goal=(c.origin_pos+c.origin_ori@c.goal_pos if c.input_ref_frame=='base' else c.goal_pos)
        error=float(np.linalg.norm(goal-c.ref_pos))
        clip=bool(np.any((torque<lo)|(torque>hi)))
        seq=scene.sequence
        row=by_sequence.get(seq)
        if row is None:
            row=dict(sequence=seq,substeps=0,torque_clipped_substeps=0,
                     min_joint_limit_margin_rad=float('inf'),min_jacobian_singular_value=float('inf'),
                     max_controller_goal_error_m=0.,max_raw_torque_nm=0.,contacts=[])
            by_sequence[seq]=row;records.append(row)
        row['substeps']+=1
        row['torque_clipped_substeps']+=int(clip)
        row['min_joint_limit_margin_rad']=min(row['min_joint_limit_margin_rad'],float(margins.min()))
        row['min_jacobian_singular_value']=min(row['min_jacobian_singular_value'],singular)
        row['max_controller_goal_error_m']=max(row['max_controller_goal_error_m'],error)
        row['max_raw_torque_nm']=max(row['max_raw_torque_nm'],float(np.abs(torque).max()))
        for contact in sim.data.contact[:sim.data.ncon]:
            names=[sim.model.geom_id2name(int(contact.geom1)),sim.model.geom_id2name(int(contact.geom2))]
            if names not in row['contacts']:row['contacts'].append(names)
        return torque

    controller.run_controller=run
    try:
        yield
    finally:
        if own:controller.run_controller=prior
        else:del controller.run_controller
