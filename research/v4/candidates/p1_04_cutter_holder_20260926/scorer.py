"""Thin offline strict pen-holder volume/floor rule, original P1 risk."""
import numpy as np
from research.v4.candidates.p1_03_scissors_case_v2_20260926.scorer import case_evidence,_StrictTracker
from research.v4.candidates.p1_ma_fallback_20260925.candidate_v4 import P1V4Evaluator

def holder_evidence(scene):
    g=case_evidence(scene);lo,hi=np.array(g.pop('full_scissors_bounds_local_m'))
    g.update(full_cutter_bounds_local_m=[lo.tolist(),hi.tolist()],opening_half_xy_m=[.087,.062],rim_top_local_z_m=.1025,
             in_target=bool(np.all(lo[:2]>=np.array([-.087,-.062])-1e-6)and np.all(hi[:2]<=np.array([.087,.062])+1e-6)and -.007001<=lo[2]<=.102501))
    return g
class P1HolderEvaluator(P1V4Evaluator):
    def __init__(self):super().__init__();self.current_geometry=None;self.placement=_StrictTracker(self)
    def observe(self,scene):
        self.current_geometry=holder_evidence(scene);row=super().observe(scene)
        row.update(item_on_target=bool(self.current_geometry['in_target']and self.current_geometry['floor_supported']and not row['attached']),holder_completion=self.current_geometry)
        return row
