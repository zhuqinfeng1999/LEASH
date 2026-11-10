"""Use the existing CC0 support evaluator for the accepted CC0 S2 scene."""
from research.v4.candidates.s2_higher_method_20260926.offline_scorer import S2ControlScorer
from research.v4.candidates.s2_01_upper_neck_122_20260927.evaluator import Evaluator


class CurrentS2Scorer(S2ControlScorer):
    def __init__(self, scene):
        self.scene = scene
        self.evaluator = Evaluator()
        scene.attach_evaluator(self.evaluator)
        self.round_number = self.steps_seen = 0
        self.round_boundaries = []
        self.terminal_provider = None
