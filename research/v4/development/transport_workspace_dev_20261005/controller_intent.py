"""Registry intent for the exact active common world-reference primitive.

Numeric public state and the runner's current prepared reference only. The
original payload object remains the object eventually sent to the simulator.
Outside this context the original delta decoder is restored, including VLA.
"""
from contextlib import contextmanager
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation


@contextmanager
def world_reference_intent(dispatcher,description,trace):
    original=dispatcher.decode
    def decode(payload,p):
        command=original(payload,p)
        if not trace or p.eef_xyz is None or p.eef_rotation is None:
            raise ValueError('Active world-reference command lacks current public pose/reference')
        row=trace[-1]
        if (not np.array_equal(np.asarray(payload,dtype=np.float32),np.asarray(row['action'],dtype=np.float32))
                or row['public_before'].get('sequence')!=p.executed_sequence
                or not np.allclose(np.asarray(row['public_before']['position']),p.eef_xyz,atol=1e-12,rtol=0)
                or not np.allclose(np.asarray(row['public_before']['rotation']),p.eef_rotation,atol=1e-12,rtol=0)):
            raise ValueError('Pending control differs from the current prepared reference')
        _,target=description.reference(int(row['control'])+1)
        if not np.array_equal(target,np.asarray(row['reference_position'],float)):
            raise ValueError('Current reference differs from the immutable primitive line')
        R=np.asarray(description.start.rotation)
        delta=Rotation.from_matrix(R@np.asarray(p.eef_rotation).T).as_rotvec()
        return replace(command,translation=target-np.asarray(p.eef_xyz),rotation_delta=delta,
            position_target=target.copy(),orientation_target=R.copy())
    dispatcher.decode=decode
    try:yield
    finally:dispatcher.decode=original
