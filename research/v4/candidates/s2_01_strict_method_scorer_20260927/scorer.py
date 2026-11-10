"""Common private S2-01 scorer for future B0/Ours runs; no online inputs."""
from research.v4.candidates.s2_higher_method_20260926.offline_scorer import (
    S2ControlScorer as OriginalControlScorer,
)
from research.v4.candidates.s2_01_strict_floor_20260927.probe import Evaluator

PROTOCOL = "S201-original-carrier-common-full-footprint-floor-v1"


class S2StrictControlScorer(OriginalControlScorer):
    """Reuse the existing per-step/round callback, with one strict evaluator.

    Installing it exactly once avoids a second observation at the same physical
    timestamp. The inherited methods remain private logging/scoring callbacks.
    This factory is identical for B0 and Ours; method identity is not an input.
    """

    def __init__(self, scene):
        self.scene = scene
        self.evaluator = Evaluator()
        scene.attach_evaluator(self.evaluator)
        self.round_number = 0
        self.steps_seen = 0
        self.round_boundaries = []
        self.terminal_provider = None
