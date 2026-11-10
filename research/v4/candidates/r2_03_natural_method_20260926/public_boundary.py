"""Only public visible nouns change; the original T1 geometry stays common."""
from unittest.mock import patch
from research.v4.candidates.r2_02_natural_method_20260926 import public_boundary as base
from research.v4.candidates.p4_natural_method_20260926 import review_worker as live
PUBLIC_NOUNS={'item':'lit tealight candle','box':'open paper shopping bag'}
ALIASES={x:'item' for x in ('item','candle','tealight','tea light','tealight candle','lit candle','lit tealight candle','burning candle')}
ALIASES.update({x:'box' for x in ('box','bag','paper bag','shopping bag','paper shopping bag','open paper shopping bag','container')})
class TealightWorkspaceReader(base.PublicWorkspaceReader):
    def __call__(self):
        with patch.object(base,'PUBLIC_NOUNS',PUBLIC_NOUNS):return super().__call__()
    def locate_vla_target(self,prompt,workspace):
        with patch.object(base,'ALIASES',ALIASES):return super().locate_vla_target(prompt,workspace)
class TealightMotionReviewer(base.PublicMotionReviewer):
    def review_motion(self,**kwargs):
        with patch.object(base,'ALIASES',ALIASES):return super().review_motion(**kwargs)
def bind_task(decision):
    with patch.object(base,'ALIASES',ALIASES):return base.bind_task(decision)
class TealightReviewerWorker(live.P4ReviewerWorker):
    def review(self,gate,packet):
        with patch.object(live,'PROMPTS',PUBLIC_NOUNS):return super().review(gate,packet)
