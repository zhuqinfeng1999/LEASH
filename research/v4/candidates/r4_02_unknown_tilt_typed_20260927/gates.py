"""R402 reporting-only reuse of the reviewed E1 typed outlet."""
import numpy as np
from research.v4.candidates.r4_02_proceed_outlet_v2_20260927.gates import Adapter as PreviousAdapter
from research.v4.candidates.e1_unknown_tilt_typed_20260927.outlet import TypedUnknownTiltOutlet, evidence_valid
from research.v4.accelerated.p3_gated_adapter import P3GatedAdapter

class Adapter(PreviousAdapter):
 def __init__(self,*args,**kwargs):
  self._prior_public_packet_valid=False
  super().__init__(*args,**kwargs)
  old=self.commit_gate
  self.commit_gate=TypedUnknownTiltOutlet(controller=old.controller,
   public_state=old.public_state,capture=old.capture,reviewer=old.reviewer,
   default_commit_conditions=old.default_commit_conditions,
   review_timeout_s=old.review_timeout_ns/1e9,clock_ns=old.clock_ns,
   packet_valid=lambda:self._prior_public_packet_valid and not self.observation_fault)
 def _sensor_packet(self,*,refresh=False):
  p=super()._sensor_packet(refresh=refresh)
  for name in ('agentview','robot0_eye_in_hand'):
   rgb=np.asarray(p['cameras'][name]['rgb'])
   if rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8:
    raise ValueError('Public RGB buffer is not native uint8 640x480')
  return p
 def capture_commit_evidence(self):
  self._prior_public_packet_valid=False
  evidence=P3GatedAdapter.capture_commit_evidence(self)
  if not evidence_valid(evidence):raise ValueError('Invalid prior public packet')
  # Empty-view evidence cannot witness the currently held unknown-axis case.
  self._prior_public_packet_valid=(evidence.robot.hold_estimate=='held')
  return evidence
