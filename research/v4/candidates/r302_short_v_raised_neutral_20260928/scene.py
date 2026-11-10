"""Only normal constructor initial-robot selection; no post-reset robot writes."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from research.v4.accelerated import tabletop
from research.v4.candidates.r3_03_t1_derived_20260926 import scene as last_override
from research.v4.candidates.r302_horizontal_short_v_20260928.scene import ShortVHorizontalCanScene
from research.v4.candidates.r302_horizontal_short_v_20260928.copy import exact_contacts

H=Path(__file__).resolve().parent
CANONICAL_RAISED_PARK=tuple(tabletop.PARK_QPOS)
EXPECTED=(0.,-.17536149667384343,0.,-1.7895761989061578,0.,1.7416096266501226,.7853981633974483)
if CANONICAL_RAISED_PARK!=EXPECTED:raise RuntimeError('Original canonical raised PARK changed')


class RaisedNeutralShortVCanScene(ShortVHorizontalCanScene):
    layout_version='R302-horizontal-short-realV-original-plus42-canonical-raised-initial-v1'
    def __init__(self,*args,**kwargs):
        before=last_override.NEUTRAL_QPOS
        if tuple(before)==CANONICAL_RAISED_PARK:
            raise RuntimeError('Original own PARK already same; do not replay equivalent constructor')
        with patch.object(last_override,'NEUTRAL_QPOS',CANONICAL_RAISED_PARK):
            super().__init__(*args,**kwargs)
        restored=last_override.NEUTRAL_QPOS is before
        sim=self.backend.sim;m,d=sim.model._model,sim.data._data
        adr=[int(m.jnt_qposadr[m.joint('robot0_joint'+str(i)).id]) for i in range(1,8)]
        robot=self.backend.robots[0];controller=robot.part_controllers['right']
        actual=d.qpos[adr].copy();init=np.asarray(robot.init_qpos);nullspace=np.asarray(controller.initial_joint)
        contacts=exact_contacts(m,d)
        bad=[x for x in contacts if x['dist_m']<-.001 and not x['exact_two_soft_pad_can_material']]
        spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec)
        np.savez_compressed(H/'initial_constructor_state.npz',integration_state=state,
                            qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),eq_active=d.eq_active.copy())
        fields={}
        for n in ('initial_joint','joint_pos','ref_pos','ref_ori_mat','initial_ref_pos','initial_ref_ori_mat',
                  'goal_pos','goal_ori','kp','kd','control_dim','input_type','input_ref_frame','position_limits','orientation_limits'):
            x=getattr(controller,n,None)
            fields[n]=x.tolist() if isinstance(x,np.ndarray) else x
        passed=bool(restored and np.array_equal(actual,CANONICAL_RAISED_PARK)
                    and np.array_equal(init,CANONICAL_RAISED_PARK)
                    and np.array_equal(nullspace,CANONICAL_RAISED_PARK) and not bad)
        report={'kind':'actual own selected normal constructor/reset/OSC admission','passed':passed,
          'old_folded_initial':list(before),'selected_canonical_raised_initial':list(CANONICAL_RAISED_PARK),
          'actual_arm_qpos':actual.tolist(),'robot_init_qpos':init.tolist(),'controller_initial_joint':nullspace.tolist(),
          'controller_type':type(controller).__module__+'.'+type(controller).__qualname__,
          'controller_fields':fields,'last_override_scope_restored':restored,'initial_contacts':contacts,
          'first_hard_contact':bad[0] if bad else None,'source_sequence':self.sequence,'sim_time_s':float(d.time),
          'extra_task_controls':0,'runtime_robot_qpos_or_controller_reset_writes':0,
          'initial_state_npz_sha256':hashlib.sha256((H/'initial_constructor_state.npz').read_bytes()).hexdigest()}
        (H/'initial_constructor_guard.json').write_text(json.dumps(report,indent=2)+'\n')
        if not passed:raise RuntimeError('Canonical selected initial/OSC/scope/contact admission failed')
