"""E2 solid-payload common controller outlet, identical to tested E1.

No actor occupancy, source pose, contact flags or evaluator state is read.
The inherited outlet intercepts held gripper-open analytic/VLA commands.
"""
from research.v4.accelerated.e1_gated_adapter import E1GatedAdapter

class E2GatedAdapter(E1GatedAdapter):
    pass
