"""Disabled-by-default R303 precompile base-forward mount proposal.

Only the original tabletop `robot.set_base_xpos` call is intercepted during
construction. This module is not imported by the current method factory.
"""
import inspect
from unittest.mock import patch
import numpy as np
from robosuite.models.robots.manipulators.panda_robot import Panda
from research.v4.candidates.r3_v42_tabletop_20260925 import scene as carrier
from research.v4.candidates.r3_03_own_horizontal_actual_scripts_20260928.entry import SelectedOwnThermosScene

ORIGINAL_SET_BASE=Panda.set_base_xpos
EXPECTED_OLD=np.array([-.61,0.,0.])
SHIFT=np.array([.09,0.,0.])
BASE_LOAD=carrier._R3FixtureEnv.__mro__[1]._load_model
CALL='robot.set_base_xpos(robot.base_xpos_offset["table"](table_size[0]))'

class BaseForward90OwnThermosScene(SelectedOwnThermosScene):
    layout_version='R303-one-common-forward-90mm-Panda-base-proposed-v1'
    def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1,
                 root_base_forward_approved=False):
        if root_base_forward_approved is not True:
            raise ValueError('R303 base-forward mounting proposal is not adopted')
        if inspect.getsource(BASE_LOAD).count(CALL)!=1:
            raise ValueError('Actual common precompile Panda mounting seam changed')
        calls=[]
        def selected_table_mount(robot_model,pos):
            old=np.asarray(pos,dtype=float)
            if old.shape==(3,) and np.allclose(old,EXPECTED_OLD,rtol=0,atol=1e-12):
                calls.append(old.copy())
                return ORIGINAL_SET_BASE(robot_model,old+SHIFT)
            return ORIGINAL_SET_BASE(robot_model,pos)
        # The original method's own robot model has not yet been merged into
        # ManipulationTask here. No compiled MjModel/MjData or episode qpos is
        # subsequently translated, and the patch is removed on constructor
        # success or exception.
        with patch.object(Panda,'set_base_xpos',selected_table_mount):
            super().__init__(seed,camera_observations=camera_observations,
                             render_gpu_device_id=render_gpu_device_id)
        if len(calls)!=1:
            self.close()
            raise ValueError('Expected exactly one selected tabletop base mount')
        m=self.backend.sim.model._model
        compiled=np.asarray(m.body_pos[m.body('robot0_base').id])
        if not np.allclose(compiled,[-.52,0.,.912],rtol=0,atol=1e-9):
            self.close()
            raise ValueError('Compiled R303 base position did not use forward mount')
