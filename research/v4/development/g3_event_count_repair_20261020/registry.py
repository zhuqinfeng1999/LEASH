"""Strict source binding around the unchanged finite G3 event/scope contract.

Every prepared control must match its immutable source index, executed
sequence, fixed gripper target and actual controller intent. A corrupt source
is a technical UNRESOLVED stop, never permission to replay an earlier action.
"""
from numbers import Integral
import numpy as np

from research.v4.development.g3_transition_dev_20261019.scope import TransitionRegistry
from research.v4.leash_v43.registry import Classification, Match

VERSION='g3-exact-source-and-completed-open-repair-dev1-20261020'


class CountBoundTransitionRegistry(TransitionRegistry):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._chunk_source_anchor=None

    def _source_problem(self,cmd,p,context):
        if self.motion is not None:
            description,trace=self.motion
            if not trace:return 'prepared source lacks its current control'
            index=trace[-1].get('control')
            start=getattr(description.start,'sequence',None)
            if (isinstance(index,bool) or not isinstance(index,Integral)
                    or not isinstance(start,Integral) or isinstance(start,bool)
                    or not 0<=index<description.max_steps
                    or p.executed_sequence!=int(start)+int(index)):
                return 'prepared source index and executed sequence are not in lockstep'
            payload=np.asarray(cmd.payload,dtype=np.float32)
            if (payload.shape!=(7,) or not np.isfinite(payload).all()
                    or payload[6]!=np.float32(description.gripper)):
                return 'prepared fixed gripper target changed'
            _,reference=description.reference(int(index)+1)
            if (cmd.position_target is None or cmd.orientation_target is None
                    or not np.allclose(cmd.position_target,reference,atol=1e-12,rtol=0)
                    or not np.allclose(cmd.orientation_target,description.start.rotation,atol=1e-12,rtol=0)):
                return 'actual candidate differs from the indexed prepared controller target'
        elif self.chunk is not None:
            identity,items,index=self.chunk
            if (isinstance(index,bool) or not isinstance(index,Integral)
                    or not 0<=index<len(items) or len(items)>5):
                return 'known generated chunk has an invalid index or exceeds five controls'
            if not np.array_equal(np.asarray(cmd.payload,dtype=np.float32),
                                  np.asarray(items[index],dtype=np.float32)):
                return 'actual candidate differs from the indexed generated control'
            primitive=(context.episode,context.round_id,context.primitive)
            anchor=self._chunk_source_anchor
            if anchor is None or anchor[0] is not identity:
                if index!=0:return 'new generated chunk must begin at its first control'
                anchor=(identity,primitive,p.executed_sequence)
                self._chunk_source_anchor=anchor
            if anchor[1]!=primitive or p.executed_sequence!=anchor[2]+int(index):
                return 'generated chunk index and executed sequence are not in lockstep'
        return None

    def _covers(self,event,cmd,p,context,permit=None):
        w=self.windows.get(event)
        permit=permit or self.permits.get(event)
        source=self._source()
        if (w is not None and permit is not None and source is not None
                and source[2]-w.first_index!=p.executed_sequence-permit.issued_sequence):
            return False
        if self.motion is not None:
            description,_=self.motion
            payload=np.asarray(cmd.payload,dtype=np.float32)
            if payload.shape!=(7,) or payload[6]!=np.float32(description.gripper):
                return False
        return super()._covers(event,cmd,p,context,permit)

    def classify(self,cmd,p,context,lease):
        problem=self._source_problem(cmd,p,context)
        if problem:
            self.invalidate()
            return Classification(Match.UNRESOLVED,(),problem)
        return super().classify(cmd,p,context,lease)
