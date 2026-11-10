"""R402 v2: public noun is not a measured held axis/tilt exemption."""
from research.v4.candidates.r4_02_proceed_outlet_20260927.gates import Adapter as PreviousAdapter
from research.v4.accelerated.p3_gated_adapter import P3GatedAdapter
class Adapter(PreviousAdapter):
 def set_public_nonpourable_payload(self,observed):
  if type(observed)is not bool:raise ValueError('Public semantic observation must be Boolean')
  # Parent workspace may establish a named cable association. That is NOT
  # a public measurement of held-object axis or permission to exempt tilt.
  self._public_nonpourable=False
 def _state_from_packet(self,packet):
  # Preserve the already-existing shared P3 public latch + unknown tilt state.
  return P3GatedAdapter._state_from_packet(self,packet)
