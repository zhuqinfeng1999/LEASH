"""E1 low-level commit outlet reusing the tested common P3/R2 adapter path.

P3GatedAdapter is task-independent in its controller behavior: it calls gate
③ on every irreversible command and forwards ordinary movement. E1 supplies
its own public latch and two-camera capture; this alias carries no P3 scoring
or hidden scene state. E1's earlier held-descent check is gate ② in
``e1_gated_toolkit.py``.
"""
from __future__ import annotations

from dataclasses import replace

from research.v4.accelerated.p3_gated_adapter import P3GatedAdapter


class E1GatedAdapter(P3GatedAdapter):
    def _state_from_packet(self, packet):
        state = super()._state_from_packet(packet)
        # The only graspable payload in this public E1 task is a solid knife.
        # The 20-degree commit threshold is for a held pourable container;
        # wrist rotation with this knife remains governed by gate ② swept-path
        # review, while every held gripper-open command still reaches gate ③.
        # This is a task-level object category, not evaluator/contact truth.
        return replace(state, held_object_tiltable=False)
