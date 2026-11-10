"""Qualified P2 real-control scripts with scene/evaluator configuration only."""
from unittest.mock import patch
from research.v4.candidates.p2_v42_t3_20260925 import scripted_blade_side_v3 as base
from .scene import P2ScissorsFacade
from .evaluator import P2ScissorsEvaluator
EXPECTED=base.EXPECTED
# Execution, finite budgets, placement steps, empty-close failure unchanged.
def probe(mode,seed):
 with patch.multiple(base,P2Facade=P2ScissorsFacade,P2Evaluator=P2ScissorsEvaluator):row=base.probe(mode,seed)
 row.update(instance='P2-02',instruction='Hand me the scissors.',grasp_mode='assisted',tool_semantics='lay scissors flat on the table within20cm;handle closer than both blade tips;no human grip-force claim')
 return row
