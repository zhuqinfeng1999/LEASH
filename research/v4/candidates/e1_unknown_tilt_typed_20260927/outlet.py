"""Typed diagnostic cause over the unchanged commit outlet; never a permit."""
from dataclasses import dataclass, replace
from enum import Enum
import numpy as np
from research.v4.candidates.leash_commit_gate_20260924.commit_gate import (
    CommitGateOutlet, GateDecision, PublicRobotState, LowLevelInstruction)
from research.v4.accelerated.p3_gated_adapter import P3GatedAdapter

class FailureKind(str, Enum):
    PUBLIC_UNKNOWN_TILT = 'public_capability_unknown_tilt'

@dataclass(frozen=True)
class TypedDecision(GateDecision):
    failure_kind: str | None = None

class HoldWitness:
    def __init__(self, controller):
        self.original=controller;self.hold_succeeded=False;self.hold_failed=False
    def hold_current_pose(self):
        try:self.original.hold_current_pose()
        except BaseException:
            self.hold_failed=True;raise
        self.hold_succeeded=True
    def send(self, instruction):return self.original.send(instruction)

class TypedUnknownTiltOutlet(CommitGateOutlet):
    def __init__(self, *, packet_valid, **kwargs):
        super().__init__(**kwargs);self.packet_valid=packet_valid
        self._typed_candidate=False
    def _classify(self, instruction, state, *, open_already_sent_for_hold=False):
        result=super()._classify(instruction,state,
            open_already_sent_for_hold=open_already_sent_for_hold)
        # Structural facts only. No Scene ID, semantic label or reason parsing.
        self._typed_candidate=bool(
            isinstance(instruction,LowLevelInstruction) and
            isinstance(state,PublicRobotState) and state.source=='public_observation' and
            state.hold_estimate=='held' and type(state.gripper_closed) is bool and
            isinstance(state.held_object_name,str) and state.held_object_name.strip() and
            state.held_object_tiltable is None and state.held_object_tilt_deg is None and
            instruction.changes_payload_orientation and
            instruction.gripper_target in (None,'closed') and
            instruction.heated_device_door_target is None and
            result[0]=='held_payload_tilt' and self.packet_valid() is True)
        return result
    def run_block(self,instructions,*,lease=None):
        original=self.controller;witness=HoldWitness(original)
        self.controller=witness;self._typed_candidate=False
        try:outcome=super().run_block(instructions,lease=lease)
        finally:self.controller=original
        rows=[]
        for index,row in enumerate(outcome.gate_decisions):
            kind=None
            if (index==len(outcome.gate_decisions)-1 and self._typed_candidate and
                witness.hold_succeeded and not witness.hold_failed and
                row.trigger=='held_payload_tilt' and row.result=='BLOCK' and
                row.system_failure and not row.reviewer_called):
                kind=FailureKind.PUBLIC_UNKNOWN_TILT.value
            rows.append(TypedDecision(**vars(row),failure_kind=kind))
        return replace(outcome,gate_decisions=tuple(rows))

def evidence_valid(evidence):
    if CommitGateOutlet._validate_evidence(evidence,0) is not None:return False
    for camera in (evidence.agentview,evidence.robot0_eye_in_hand):
        if (camera.width,camera.height)!=(640,480):return False
        k=np.asarray(camera.intrinsics.get('K'),float)
        t=np.asarray(camera.extrinsics.get('camera_to_world'),float)
        if k.shape!=(3,3) or t.shape!=(4,4) or not np.isfinite(k).all() or not np.isfinite(t).all():return False
        if k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1]):return False
        if not np.allclose(t[3],[0,0,0,1]) or abs(np.linalg.det(t[:3,:3])-1)>1e-4:return False
    return True

class TypedContainerAdapter(P3GatedAdapter):
    def __init__(self,*args,**kwargs):
        self._prior_public_packet_valid=False
        super().__init__(*args,**kwargs)
        old=self.commit_gate
        self.commit_gate=TypedUnknownTiltOutlet(
            controller=old.controller,public_state=old.public_state,capture=old.capture,
            reviewer=old.reviewer,default_commit_conditions=old.default_commit_conditions,
            review_timeout_s=old.review_timeout_ns/1e9,clock_ns=old.clock_ns,
            packet_valid=lambda:self._prior_public_packet_valid and not self.observation_fault)
    def _sensor_packet(self,*,refresh=False):
        packet=super()._sensor_packet(refresh=refresh)
        for name in ('agentview','robot0_eye_in_hand'):
            rgb=np.asarray(packet['cameras'][name]['rgb'])
            if rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8:
                raise ValueError('Public RGB buffer is not native uint8 640x480')
        return packet
    def capture_commit_evidence(self):
        self._prior_public_packet_valid=False
        result=super().capture_commit_evidence()
        self._prior_public_packet_valid=evidence_valid(result)
        if not self._prior_public_packet_valid:raise ValueError('Invalid prior public packet')
        return result

def commit_infrastructure_failure(rows):
    """Only an outlet-issued typed cause can distinguish legacy raw fault flags."""
    return any(row.get('system_failure') and not (
        row.get('failure_kind')==FailureKind.PUBLIC_UNKNOWN_TILT.value and
        row.get('gate')==3 and row.get('boundary')=='executor_to_controller' and
        row.get('trigger')=='held_payload_tilt' and row.get('result')=='BLOCK' and
        row.get('reviewer_called') is False) for row in rows)
