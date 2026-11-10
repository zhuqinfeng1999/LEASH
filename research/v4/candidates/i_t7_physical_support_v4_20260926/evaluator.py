"""Future common I completion revision; hazards/critical and physics inherited."""
from research.v4.candidates.i_t7_named_script_strict_v3_20260926.evaluator import Evaluator as Original
from .support import PhysicalBasketTracker
class Evaluator(Original):
    def __init__(self,instance):
        super().__init__(instance);self.current_scene=None
        self.placement=PhysicalBasketTracker(lambda:self.current_scene)
    def observe(self,scene):
        self.current_scene=scene;state=super().observe(scene)
        state['private_placement_geometry']=self.placement.last_geometry
        return state
