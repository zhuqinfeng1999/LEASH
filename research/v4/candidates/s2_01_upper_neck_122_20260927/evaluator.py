"""Same S2 state machine; root-approved exact-cylinder floor alias only."""
from unittest.mock import patch
from research.v4.candidates.s2_template_floor_completion_20260926 import evaluator as parent
from .support import PhysicalBasketTracker
class Evaluator(parent.Evaluator):
    def begin_round(self,*args,**options):
        with patch.object(parent,'PhysicalBasketTracker',PhysicalBasketTracker):
            return super().begin_round(*args,**options)
