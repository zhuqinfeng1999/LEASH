"""Thin offline target/support substitution; all other S3 scoring inherited."""
from research.v4.accelerated.scoring import PlacementTracker
from research.v4.candidates.s3_v42_tabletop_20260926.offline_scorer import S3ControlScorer
from research.v4.candidates.s3_rest_diagnostics_20260926.private_diagnostics import snapshot
from .predicate import basket_evidence,REVISION


class PhysicalBasketTracker(PlacementTracker):
    def __init__(self,scene_provider):
        super().__init__();self._scene_provider=scene_provider;self.last_geometry=None
    def observe(self,**kwargs):
        self.last_geometry=basket_evidence(self._scene_provider())
        kwargs['in_target']=bool(self.last_geometry['in_target'])
        kwargs['supported']=bool(self.last_geometry['floor_supported'])
        return super().observe(**kwargs)


class PhysicalSupportS3Scorer(S3ControlScorer):
    def __init__(self):
        super().__init__();self._current_scene=None
        self.ketchup_tracker=PhysicalBasketTracker(lambda:self._current_scene)
    def observe(self,scene):
        self._current_scene=scene
        state=super().observe(scene)  # Exactly once; original plate/cookie/evaluator.
        diagnostic=snapshot(scene,self);k=diagnostic['ketchup'];g=self.ketchup_tracker.last_geometry
        k['legacy_origin_box_diagnostic_only']=k['target'];k['target']=g
        k['floor_support_evidence']=g['floor_contacts']
        k['rest_requirements']['in_target']=bool(g['in_target'])
        k['rest_requirements']['named_support']=bool(g['floor_supported'])
        k['rest_requirements_all_now']=all(k['rest_requirements'].values())
        k['rest_failures_now']=[n for n,v in k['rest_requirements'].items()if not v]
        diagnostic['revision']=REVISION;state['private_rest_diagnostics']=diagnostic
        return state
