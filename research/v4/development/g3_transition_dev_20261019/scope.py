"""Finite reviewed continuation, using public commands and proprioception only.

This does not decide safety, generate motion, classify objects, or change a
holding estimate. A window becomes usable only via the existing PASS->grant
dispatch path. Unprepared adaptive controls retain the original registry.
"""
from contextlib import contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math

import numpy as np

from research.v4.leash_v43.registry import (
    CommitEventRegistry, Match, Holding, RELEASE, POUR, axis_rotation,
)
from research.v4.leash_v43.dispatch import CommitDispatcher

VERSION = 'finite-reviewed-commit-transition-dev1-20261019'


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def angle(a,b):
    return math.acos(float(np.clip((np.trace(np.asarray(a)@np.asarray(b).T)-1)/2,-1,1)))


def line_distance(point, points):
    point=np.asarray(point);points=np.asarray(points)
    if len(points)==1:return float(np.linalg.norm(point-points[0]))
    a,b=points[:-1],points[1:];v=b-a
    denom=np.sum(v*v,axis=1)
    t=np.divide(np.sum((point-a)*v,axis=1),denom,
                out=np.zeros_like(denom),where=denom>0).clip(0,1)
    return float(np.min(np.linalg.norm(point-a-t[:,None]*v,axis=1)))


@dataclass
class ReviewedWindow:
    descriptor: dict
    identity: object
    generation: int
    context: object
    first_index: int
    last_index: int
    expires_sequence: int
    gripper: str
    rotation_anchor: np.ndarray
    orientation_radius: float
    points: np.ndarray
    payloads: tuple
    grasp_digest: str | None


class TransitionRegistry(CommitEventRegistry):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.windows={}
        self.pending_window=None
        self.motion=None
        self.chunk=None
        self.window_logs=[]
        self._saved_permits={}

    def invalidate(self):
        super().invalidate()
        self.windows.clear()
        self.pending_window=None

    def _source(self):
        if self.motion is not None:
            description,trace=self.motion
            if not trace:return None
            return ('prepared_common_reference',description,int(trace[-1]['control']),trace)
        if self.chunk is not None:
            identity,items,index=self.chunk
            return ('known_control_chunk',identity,index,items)
        return None

    def _covers(self,event,cmd,p,context,permit=None):
        w=self.windows.get(event)
        permit=permit or self.permits.get(event)
        source=self._source()
        if (w is None or permit is None or source is None or p.eef_xyz is None
                or p.eef_rotation is None or cmd.position_target is None
                or cmd.orientation_target is None or w.identity is not source[1]
                or w.context!=context or permit.context!=context
                or w.generation!=self.estimator.generation
                or permit.generation!=self.estimator.generation
                or not w.first_index<=source[2]<=w.last_index
                or not permit.issued_sequence<=p.executed_sequence<=min(w.expires_sequence,permit.expires_sequence)
                or cmd.gripper!=w.gripper):
            return False
        # A later actual CLOSE phase cannot inherit an earlier possible-pour
        # authorization. OPEN does not create a fictitious EMPTY observation.
        current_grasp=None if self.estimator.grasp_rotation is None else digest(self.estimator.grasp_rotation.tolist())
        if event==POUR and current_grasp!=w.grasp_digest:return False
        if source[0]=='prepared_common_reference':
            description=source[1]
            _,reference=description.reference(source[2]+1)
            if (not np.allclose(cmd.position_target,reference,atol=1e-12,rtol=0)
                    or not np.allclose(cmd.orientation_target,np.asarray(description.start.rotation),atol=1e-12,rtol=0)):
                return False
            spatial=all(line_distance(v,w.points)<=self.position_tolerance_m for v in (p.eef_xyz,cmd.position_target))
        else:
            offset=source[2]-w.first_index
            if not np.array_equal(np.asarray(cmd.payload,dtype=np.float32),np.asarray(w.payloads[offset],dtype=np.float32)):
                return False
            lo=w.points.min(axis=0)-self.position_tolerance_m
            hi=w.points.max(axis=0)+self.position_tolerance_m
            spatial=all(np.all(v>=lo) and np.all(v<=hi) for v in (p.eef_xyz,cmd.position_target))
        return bool(spatial and all(angle(v,w.rotation_anchor)<=w.orientation_radius+1e-10
                                    for v in (p.eef_rotation,cmd.orientation_target)))

    def _valid_permit(self,event,cmd,p,context,direction=None):
        if event in self.windows:
            return self._covers(event,cmd,p,context)
        return super()._valid_permit(event,cmd,p,context,direction)

    def _proposal(self,cmd,p,context):
        source=self._source()
        if (source is None or p.eef_xyz is None or p.eef_rotation is None
                or cmd.position_target is None or cmd.orientation_target is None):return None
        kind,identity,index,extra=source
        current=np.asarray(p.eef_xyz)
        if kind=='prepared_common_reference':
            description=identity
            if cmd.gripper!=('OPEN' if description.gripper<0 else 'CLOSE' if description.gripper>0 else 'HOLD'):
                return None
            _,reference=description.reference(index+1)
            if not np.allclose(reference,cmd.position_target,atol=1e-12,rtol=0):return None
            points=[current.tolist()]
            for i in range(index,description.max_steps):
                _,point=description.reference(i+1)
                if point.tolist()!=points[-1]:points.append(point.tolist())
            anchor=np.asarray(description.start.rotation)
            radius=min(math.pi,angle(p.eef_rotation,anchor)+self.direction_tolerance_rad)
            payloads=()
            last_index=description.max_steps-1
            detail=dict(prepared_reference_sha256=digest(description.as_dict()),
                reference_points_world_m=points,fixed_commanded_rotation=anchor.tolist(),
                position_scope='within existing 8mm tracking tolerance of these reviewed reference segments',
                remaining_controls=max(1,description.max_steps-index))
        else:
            items=extra
            prefix=[]
            for item in items[index:]:
                # Never authorize a later closing/opening cycle as continuation.
                value=float(np.asarray(item)[6])
                sign='OPEN' if value<0 else 'CLOSE' if value>0 else 'HOLD'
                if sign!=cmd.gripper:break
                prefix.append(np.asarray(item,dtype=np.float32).copy())
            if not prefix or len(prefix)>5:return None
            if not np.array_equal(prefix[0],np.asarray(cmd.payload,dtype=np.float32)):return None
            # Unroll only already-generated controls with the same installed
            # public decoder; this is a commanded scope, not object prediction.
            decoded=[];position=current.copy();rotation=np.asarray(p.eef_rotation).copy()
            points=[position.tolist()];rotation_sum=0.
            for item in prefix:
                state=replace(p,eef_xyz=position,eef_rotation=rotation)
                c=self.chunk_decoder(item,state)
                decoded.append(dict(control=item.tolist(),translation_m=c.translation.tolist(),
                                    rotation_world_axis_angle_rad=c.rotation_delta.tolist()))
                position=c.position_target.copy();rotation=c.orientation_target.copy()
                rotation_sum+=float(np.linalg.norm(c.rotation_delta));points.append(position.tolist())
            anchor=np.asarray(p.eef_rotation).copy()
            radius=min(math.pi,rotation_sum+self.direction_tolerance_rad)
            payloads=tuple(prefix);last_index=index+len(prefix)-1
            detail=dict(already_generated_controls=decoded,
                commanded_prefix_points_world_m=points,
                position_scope='axis-aligned extent of the disclosed generated prefix plus existing 8mm tolerance; exact payload/index binding also required',
                remaining_controls=len(prefix))
        expiry=p.executed_sequence+detail['remaining_controls']-1
        descriptor=dict(version=VERSION,source=kind,events='only the currently reviewed registered events',
            first_sequence=p.executed_sequence,last_sequence=expiry,
            gripper=cmd.gripper,**detail,
            rotation_anchor=anchor.tolist(),max_SO3_distance_from_anchor_rad=radius,
            holding_generation=self.estimator.generation,
            context=dict(episode=context.episode,round_id=context.round_id,primitive=context.primitive,
                         binding_version=context.binding_version),
            limitations='Command/reference bounds for reviewer scope only; not a safety proof, guaranteed tracking, or a permission for a future primitive. Out-of-scope control requires fresh review. No object state inferred.')
        return ReviewedWindow(descriptor,identity,self.estimator.generation,context,index,last_index,
            expiry,cmd.gripper,anchor,radius,np.asarray(points),payloads,
            None if self.estimator.grasp_rotation is None else digest(self.estimator.grasp_rotation.tolist()))

    def classify(self,cmd,p,context,lease):
        self.pending_window=None
        # Closing starts/restarts a different release phase even if thin
        # objects leave the public holding generation unchanged.
        if cmd.gripper=='CLOSE':self.windows.pop(RELEASE,None)
        old=dict(self.permits)
        rearm=self._known_tilted
        result=super().classify(cmd,p,context,lease)
        if rearm and not self._known_tilted:self.windows.pop(POUR,None)
        if result.state==Match.UNRESOLVED or self.estimator.kind==Holding.EMPTY:return result
        matched=list(result.events);continued=list(result.permit_events)
        for event in list(matched):
            # Base release matching intentionally discards stationary permits
            # on motion. Restore only the exact existing PASS permission whose
            # explicitly reviewed continuation still covers this command.
            if self._covers(event,cmd,p,context,old.get(event)):
                self.permits[event]=old[event]
                matched.remove(event);continued.append(event)
                self.window_logs.append(dict(kind='reviewed_window_reuse',event=event,
                    sequence=p.executed_sequence,primitive=context.primitive,
                    window_sha256=digest(self.windows[event].descriptor)))
        for event in result.permit_events:
            if event in self.windows:
                self.window_logs.append(dict(kind='reviewed_window_reuse',event=event,
                    sequence=p.executed_sequence,primitive=context.primitive,
                    window_sha256=digest(self.windows[event].descriptor)))
        if matched:
            self.pending_window=self._proposal(cmd,p,context)
        return replace(result,state=Match.MATCH if matched else Match.NO_MATCH,
            events=tuple(matched),permit_events=tuple(dict.fromkeys(continued)),
            reason='registered transition requires review' if matched else 'no new transition or valid finite reviewed continuation')

    def grant(self,events,cmd,p,context,evidence_version):
        super().grant(events,cmd,p,context,evidence_version)
        proposal=self.pending_window
        for event in events:self.windows.pop(event,None)
        if proposal is not None:
            if (proposal.context!=context or proposal.generation!=self.estimator.generation
                    or proposal.descriptor['first_sequence']!=p.executed_sequence):
                raise ValueError('Reviewed continuation changed during review')
            for event in events:self.windows[event]=proposal
            self.window_logs.append(dict(kind='reviewed_window_granted',events=list(events),
                sequence=p.executed_sequence,primitive=context.primitive,
                window_sha256=digest(proposal.descriptor),window=proposal.descriptor))
        self.pending_window=None


class TransitionDispatcher(CommitDispatcher):
    def run_chunk(self,payloads):
        items=list(payloads);identity=object();output=[]
        try:
            for i,payload in enumerate(items):
                self._chunk_remaining=len(items)-i-1
                self.registry.chunk=(identity,items,i)
                # Original delta decoder outside prepared analytic context.
                self.registry.chunk_decoder=self.decode
                output.append(self.step(payload))
            return output
        finally:
            self.registry.chunk=None
            self._chunk_remaining=0
            items.clear()


@contextmanager
def prepared_motion_scope(original,dispatcher,description,trace):
    registry=dispatcher.registry
    if not isinstance(registry,TransitionRegistry):
        with original(dispatcher,description,trace):yield
        return
    if registry.motion is not None:raise RuntimeError('Nested prepared reference')
    registry.motion=(description,trace)
    try:
        with original(dispatcher,description,trace):yield
    finally:
        registry.motion=None
