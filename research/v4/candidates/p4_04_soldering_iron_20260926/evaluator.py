"""No new evaluator algorithm: hot shaft/tip vs diagram; actual cold support."""
from unittest.mock import patch
from research.v4.candidates.p4_03_gluegun_20260926 import evaluator as common
from research.v4.candidates.p4_support_scoring_20260926.evaluator import P4SupportEvaluator
HOT=('hot_shaft','heated_tip');PAPER=('circuit_paper',);SUPPORT=('insulated_handle','cold_front_collar')
def snapshot(scene):
 with patch.multiple(common,HOT=HOT,PAPER=PAPER,SUPPORT=SUPPORT):return common.snapshot(scene)
class SolderEvaluator(P4SupportEvaluator):
 def observe(self,scene):return self.observe_row(snapshot(scene),release_started=scene.last_release_started,initial_bottom_z=scene.initial_item_bottom_z)
