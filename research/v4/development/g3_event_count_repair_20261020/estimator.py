"""Retain a verified completed opening while current public evidence agrees.

An OPEN command or a reviewer PASS cannot establish EMPTY. The original
calibration and stable executed samples must establish it first. Once open
completion is verified, aperture settling above the same completion threshold
does not turn an open gripper into possible holding. A new CLOSE, telemetry
contradiction, or uncertain consumption revokes this fact immediately.
"""
import math

from research.v4.leash_v43.registry import Holding, HoldingEstimator

VERSION = 'completed-open-evidence-continuity-dev1-20261020'


class CompletedOpenHoldingEstimator(HoldingEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._completed_open_verified = False

    def observe(self, p):
        previous_sequence = self._sequence
        prior_completed_open = (self._completed_open_verified
                                and self.kind == Holding.EMPTY
                                and self.last_gripper == 'OPEN')
        super().observe(p)
        agrees = (p.aperture_m is not None and math.isfinite(p.aperture_m)
                  and p.aperture_m >= self.calibration.open_complete_gap_m
                  and self.last_gripper == 'OPEN')
        # A contradictory same-frame read is not a new stability sample, but
        # must still revoke the completed-opening fact before admission.
        if prior_completed_open and not agrees:
            self.kind = Holding.UNKNOWN
            self._samples.clear()
        if prior_completed_open and agrees and p.executed_sequence > previous_sequence:
            self.kind = Holding.EMPTY
            self._grip_active = False
            self.grasp_rotation = None
            self._executed_close_rotation = None
        self._completed_open_verified = bool(
            agrees and self.kind == Holding.EMPTY and not self._initial_empty_active)

    def record_command(self, gripper, at_execution=None):
        if gripper == 'CLOSE':
            self._completed_open_verified = False
        return super().record_command(gripper, at_execution)

    def execution_uncertain(self):
        self._completed_open_verified = False
        return super().execution_uncertain()
