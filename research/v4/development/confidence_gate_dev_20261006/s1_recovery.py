"""Runner-only S1 recovery through current shared primitive gates.

The old common recovery is administrative rather than a planner primitive:
its controls and reviews are retained, while planner call/round counters and
the original final text stay unchanged. No latch, private pose, reset or
default occupancy is consulted. This scope is never an advertised MCP tool.
"""
from contextlib import contextmanager
from copy import deepcopy
import threading
import numpy as np


class _RecoveryToolkit:
    def __init__(self, base):
        self._original_base=base

    def __getattr__(self,name):
        return getattr(self._original_base,name)

    def execute_tool(self,name,arguments):
        if name not in ('move_to','set_gripper'):
            raise PermissionError('Only the two original S1 recovery primitives are available')
        # Original RPent handler/state recording, behind the normal confidence
        # boundary and existing adapter/registry. B0 finish/call accounting is
        # intentionally not replaced with a fake additional planner call.
        return self._original_base.original.execute_tool(name,arguments)


def _accounting(base):
    return dict(round_finished=base.round_finished,finish_result=deepcopy(base.finish_result),
        round_calls=base.round_calls,episode_calls=base.episode_calls,calls=len(base.calls))


@contextmanager
def recovery_scope(stack, *, prior_seal):
    if threading.get_ident()!=stack.owner.owner_thread_id:
        raise RuntimeError('S1 recovery must execute inside the actual scene Owner callback')
    if prior_seal not in ('finish','round_call_quota','episode_call_quota',
                           'planner_turn_quota','planner_time_quota'):
        raise ValueError('No original finish/quota seal permits inter-round recovery')
    base=stack.base
    before=_accounting(base)
    runner_toolkit=_RecoveryToolkit(base)
    if stack.arm=='B0':
        yield runner_toolkit
    else:
        boundary=stack.boundary
        if boundary.base is not base:
            raise RuntimeError('S1 recovery changed its original toolkit binding')
        old_open=boundary.round_open
        boundary.base=runner_toolkit
        # The original completed G1 decision and denial/system-failure state
        # remain active. REFUSE still makes every non-finish call unavailable.
        # This permits only this trusted callback to submit a recovery action;
        # no new G1, lease, holding state or reset fact is issued.
        boundary.round_open=True
        try:
            yield boundary
        finally:
            boundary.base=base
            boundary.round_open=old_open
    if _accounting(base)!=before:
        raise RuntimeError('Administrative recovery changed planner finish or primitive accounting')


def make_recovery_executor(stack, *, seal_reader):
    if not callable(seal_reader):
        raise ValueError('Existing task driver must supply its actual prior seal')
    def execute(intent,metadata):
        if intent not in ('maintain_closed_hold','return_empty_gripper_to_public_start'):
            raise ValueError('Unknown original S1 recovery intent')
        scene=stack.scene
        sequence=int(scene.sequence)
        estimator=getattr(stack.adapter,'estimator',None)
        holding=(metadata.get('holding_estimate') if estimator is None else
                 getattr(estimator.kind,'value',estimator.kind))
        if metadata.get('executed_sequence')!=sequence or metadata.get('holding_estimate')!=holding:
            raise ValueError('Recovery public state changed before the actual callback')
        if intent=='maintain_closed_hold':
            if holding not in ('UNKNOWN','HELD'):
                raise ValueError('Closed recovery HOLD requires possible current occupancy')
            # Exact original safe-HOLD numeric command: zero motion, +1
            # gripper, one control. It is represented by the existing tool.
            name,arguments='set_gripper',dict(gripper=1.,steps=1)
        else:
            if holding!='EMPTY':
                raise ValueError('Open observation park requires current supported EMPTY')
            target=np.asarray(metadata['public_start_eef_xyz_m'],float)
            if target.shape!=(3,) or not np.isfinite(target).all():
                raise ValueError('Original public park target unavailable')
            # Preserve the old park timing/bounds. The same already-installed
            # common controller is used by B0 and Ours; no automatic descent,
            # grasp, new route, extra opening transition or reset is inserted.
            name,arguments='move_to',dict(xyz=target.tolist(),max_steps=260,
                                          gripper=-1.,step_clip=.010,tol=.009,action_scale=.05)
        with recovery_scope(stack,prior_seal=seal_reader()) as toolkit:
            result=toolkit.execute_tool(name,arguments)
        after=int(scene.sequence)
        public=getattr(result,'result',None)
        failed=not isinstance(public,dict) or bool(public.get('error')) or bool(public.get('system_failure'))
        if intent=='maintain_closed_hold':
            complete=not failed and after-sequence==1
        else:
            current=np.asarray(scene.raw['robot0_eef_pos'],float)
            complete=(not failed and current.shape==(3,) and np.isfinite(current).all()
                      and float(np.linalg.norm(current-target))<=.022)
        # RPent already saves these images under the result's step/artifacts.
        # Keep their ordinary metadata in the recovery log, not duplicate
        # in-memory PNG payloads in rounds.json. The actual tool reply above
        # and the planner's observation/history remain unchanged.
        recorded_public = (None if not isinstance(public,dict) else
            {key:deepcopy(value) for key,value in public.items()
             if key not in ('_image_bytes','_image_cam_bytes','_image_wrist_bytes')})
        return dict(complete=bool(complete),actual_control_steps=after-sequence,
            shared_tool_and_gates=True,primitive=name,arguments=arguments,
            result=recorded_public,holding_estimate_before=holding,
            planner_accounting_preserved=True,reset_or_empty_initialization=False)
    return execute
