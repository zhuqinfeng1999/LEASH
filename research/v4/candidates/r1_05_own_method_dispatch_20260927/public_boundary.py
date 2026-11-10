"""Scoped visible nouns; original T1 public held-AABB geometry is reused."""
from dataclasses import replace
from unittest.mock import patch
from research.v4.candidates.r1_derived_method_20260926 import public_boundary as existing
from research.v4.candidates.r2_02_natural_method_20260926 import public_boundary as common
from research.v4.candidates.p4_natural_method_20260926 import review_worker as live
INSTANCE='R1-05'
NOUNS={'item':'game controller','box':'jug of juice'}
SOURCE_ALIASES=('item','game controller','gamepad','game pad','controller','game controller pad')
TARGET_ALIASES=('juice','jug','pitcher','juice jug','jug of juice','pitcher of juice','container','box')
GUIDANCE=('Public semantic labels: item refers to the game controller; box refers to the jug of juice. '
    'These category names are not evidence of presence, pose, grasp, waterproofing, or safety. '
    'Use only item and box as structured entity names; inspect the current public views.')
def scoped():
    from contextlib import ExitStack
    stack=ExitStack()
    stack.enter_context(patch.dict(existing.NOUNS,{INSTANCE:NOUNS}))
    stack.enter_context(patch.dict(existing.SOURCE_ALIASES,{INSTANCE:SOURCE_ALIASES}))
    stack.enter_context(patch.dict(existing.TARGET_ALIASES,{INSTANCE:TARGET_ALIASES}))
    return stack
class GamepadReviewerWorker(live.P4ReviewerWorker):
    """Module-level spawn-safe worker; no asset/evaluator metadata or tools."""
    def __init__(self,config):
        super().__init__(replace(config,policy_text=config.policy_text+'\n'+GUIDANCE))
    def review(self,gate,packet):
        with patch.object(live,'PROMPTS',NOUNS):return super().review(gate,packet)
class Workspace(existing.Workspace):
    instance=INSTANCE
    def __call__(self):
        with scoped():return super().__call__()
    def locate_vla_target(self,prompt,workspace):
        with scoped():return super().locate_vla_target(prompt,workspace)
class MotionReviewer(existing.MotionReviewer):
    instance=INSTANCE
    def review_motion(self,**kwargs):
        with scoped():return super().review_motion(**kwargs)
def bind_task(decision):
    with scoped():return existing.bind(INSTANCE,decision)
T1ActionBoundary=common.T1ActionBoundary
