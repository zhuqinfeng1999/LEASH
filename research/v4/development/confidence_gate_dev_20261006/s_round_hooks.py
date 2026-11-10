"""Owner-side accepted S round transitions and explicit recovery seams.

Private scorers stay on the runner side. This module sends no control, reads
no grasp latch, and initializes no occupancy state. Necessary S1 recovery is
delegated to an explicitly supplied shared-tool executor, or reported pending.
"""
from copy import deepcopy
import threading
import numpy as np


class RecoveryPending(RuntimeError):
    def __init__(self, metadata):
        self.metadata = metadata
        super().__init__(metadata['reason'])


def _xyz(scene):
    value=np.asarray(scene.raw['robot0_eef_pos'],float)
    if value.shape!=(3,) or not np.isfinite(value).all():
        raise ValueError('Current public EEF position unavailable')
    return value.copy()


class SRoundHooks:
    def __init__(self, stack, configured, evaluator, *, holding_reader=None,
                 recovery_executor=None):
        if not configured.persistent:
            raise ValueError('S round hooks require registered persistent instructions')
        self.stack,self.configured,self.evaluator=stack,configured,evaluator
        self.family=configured.binding.instance.split('-')[0]
        if self.family not in ('S1','S2','S3'):
            raise ValueError('Only accepted S1/S2/S3 transitions are provided')
        self.owner_thread_id=threading.get_ident()
        self.scene_identity=id(stack.scene)
        self.registry_identity=id(getattr(stack.adapter,'registry',None))
        self.initial_eef=_xyz(stack.scene)
        self.holding_reader=holding_reader
        self.recovery_executor=recovery_executor
        self.started=0
        self.seals=[]
        self.recovery=[]
        if self.family=='S2':
            # This original scorer attaches the actual private per-control
            # evaluator in its constructor, before the first command.
            if getattr(evaluator,'scene',None) is not stack.scene:
                raise ValueError('CurrentS2Scorer must be constructed with this exact Scene')
            evaluator.terminal_provider=lambda:stack.base.finish_result

    def _owner(self):
        if (threading.get_ident()!=self.owner_thread_id or id(self.stack.scene)!=self.scene_identity
                or id(getattr(self.stack.adapter,'registry',None))!=self.registry_identity):
            raise RuntimeError('S transition changed owner, physical Scene or registry')

    def _holding(self):
        if self.holding_reader is not None:
            value=self.holding_reader()
        else:
            estimator=getattr(self.stack.adapter,'estimator',None)
            value='UNKNOWN' if estimator is None else estimator.kind
        value=getattr(value,'value',value)
        if value not in ('EMPTY','UNKNOWN','HELD'):
            raise ValueError('Only the current permitted proprio holding state is accepted')
        return value

    def _s1_recover(self):
        scene=self.stack.scene
        kind=self._holding()
        current=_xyz(scene)
        before=int(scene.sequence)
        distance=float(np.linalg.norm(current-self.initial_eef))
        if kind=='EMPTY' and distance<=.009:
            row=dict(kind='no_motion_already_at_public_start',actual_control_steps=0,
                holding_estimate=kind,public_start_eef_xyz_m=self.initial_eef.tolist(),
                current_public_eef_xyz_m=current.tolist(),executed_sequence=before,
                state_was_reinitialized=False)
            self.recovery.append(row)
            return
        intent='maintain_closed_hold' if kind in ('HELD','UNKNOWN') else 'return_empty_gripper_to_public_start'
        metadata=dict(kind='s1_common_recovery_pending',holding_estimate=kind,
            intended_existing_recovery=intent,executed_sequence=before,
            public_start_eef_xyz_m=self.initial_eef.tolist(),current_public_eef_xyz_m=current.tolist(),
            reason=('S1 recovery needs the original conservative closed HOLD through the current shared '
                    'tool/gates; finalized tool boundary currently requires an explicit administrative recovery seam.'
                    if kind in ('HELD','UNKNOWN') else
                    'S1 empty observation park needs the existing shared primitive through current G2/registry; '
                    'finalized tool boundary currently requires an explicit administrative recovery seam.'))
        if self.recovery_executor is None:
            raise RecoveryPending(metadata)
        # The trusted Root supplies the previously implemented tool execution
        # seam. No hidden Scene.step or callback-defined grasp inference exists
        # here. A callback failure stops transition and is never retried here.
        outcome=self.recovery_executor(intent,deepcopy(metadata))
        if not isinstance(outcome,dict) or outcome.get('complete') is not True:
            metadata['execution_outcome']=deepcopy(outcome)
            raise RecoveryPending(metadata)
        after=int(scene.sequence)
        count=outcome.get('actual_control_steps')
        if type(count) is not int or count<0 or after-before!=count:
            raise RuntimeError('Recovery receipt does not match actual public execution sequence')
        if outcome.get('shared_tool_and_gates') is not True:
            raise RuntimeError('Recovery did not use the existing shared tool/gate execution seam')
        self.recovery.append(dict(metadata,kind='s1_common_recovery_executed',
                                  actual_control_steps=count,outcome=deepcopy(outcome)))

    def prepare_physical_round(self,index,instruction,previous):
        self._owner()
        if (index!=self.started+1 or index>len(self.configured.instructions)
                or instruction!=self.configured.instructions[index-1]):
            raise ValueError('S instruction order or registered text changed')
        if index>1 and previous not in ('finish','round_call_quota','episode_call_quota',
                                       'planner_turn_quota','planner_time_quota'):
            raise ValueError('Prior S round has no original finish/quota seal')
        scene=self.stack.scene
        if self.family=='S1':
            if index==1:
                if scene.round_index!=0:raise ValueError('S1 did not start in original round zero')
            else:
                if scene.round_index!=index-2:raise ValueError('S1 physical round changed')
                self._s1_recover()
                scene.advance_round()
            if getattr(self.evaluator,'round_number',0)==index-1:
                self.evaluator.start_round(index,scene)
            elif self.evaluator.round_number!=index:
                raise ValueError('S1 private scorer has a different round')
        else:
            if index==1:
                if scene.round_number!=1:raise ValueError('S scene construction did not start round one')
            else:
                if scene.round_number!=index-1:raise ValueError('S physical round changed')
                scene.advance_round()
            if getattr(self.evaluator,'round_number',0)==index-1:
                self.evaluator.begin_round(index)
            elif self.evaluator.round_number!=index:
                raise ValueError('S private scorer has a different round')
        self.started=index
        return instruction

    def after_round(self,index,terminal,seal):
        self._owner()
        if index!=self.started or index!=len(self.seals)+1:
            raise ValueError('S seal order changed')
        self.seals.append(dict(index=index,terminal=deepcopy(terminal),seal=seal,
                               executed_sequence=int(self.stack.scene.sequence)))


def make_hooks(stack,configured,*,evaluator=None,holding_reader=None,recovery_executor=None):
    if evaluator is None:
        evaluator=(configured.Evaluator(stack.scene) if configured.evaluator_requires_scene
                   else configured.Evaluator())
    return SRoundHooks(stack,configured,evaluator,holding_reader=holding_reader,
                       recovery_executor=recovery_executor)
