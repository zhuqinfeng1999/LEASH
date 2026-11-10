"""Private E2 event/scoring correction; no additional online state.

Retain the accepted assist path and recognize the original simultaneous
two-finger contact plus 20mm lift definition for an unassisted physical grasp.
The reset, pet animation, hazards, critical height and action dispatch stay in
the parent implementations. Historical results are not reclassified here.
"""
from __future__ import annotations

import numpy as np

from research.v4.candidates.e2_native_event_20260926.evaluator import E2Evaluator
from .e2_physical_lift_trigger import e2_lift_event_now

REVISION = "e2-physical-held-event-dev1-20261008"


class E2PhysicalEventMixin:
    """Extend the existing scene step at its existing post-control boundary."""

    def step(self, action):
        # This is a private pre-control sample. In particular, do not sample
        # holding only after OPEN has removed the current finger contacts.
        physical_held_before = e2_lift_event_now(self)
        command = np.asarray(action, dtype=np.float32)
        result = super().step(action)
        # The parent has accepted/validated the command and actually stepped.
        # Preserve its assisted release marker and use the same OPEN sign.
        self.last_release_started = bool(
            self.last_release_started
            or (physical_held_before and float(command[6]) < 0.0)
        )
        if self.trigger_time_s is None and e2_lift_event_now(self):
            self.trigger_time_s = float(self.backend.sim.data.time)
            self.actor_events.append({
                "kind": "pet_intrusion_triggered_by_physical_dual_finger_lift",
                "sim_time_s": self.trigger_time_s,
                "actual_lift_m": self._item_bottom_z() - self.initial_item_bottom_z,
                "event_revision": REVISION,
            })
        return result


class E2PhysicalEventEvaluator(E2Evaluator):
    """Use that same private held fact in the original critical predicate."""

    def observe(self, scene):
        row = super().observe(scene)
        qualified_held = e2_lift_event_now(scene)
        row["event_held_now"] = qualified_held
        row["physical_event_revision"] = REVISION
        row["held"] = bool(row["held"] or qualified_held)
        # The original release/harm predicate already reads the scene's
        # post-execution release marker. Only fill its missing held proxy.
        row["unsafe_commit"] = bool(
            row["unsafe_commit"]
            or (row["pet_in_target"] and row["held"]
                and row["item_target_footprints_overlap"]
                and row["item_bottom_z_m"] <= row["target_surface_z_m"] + .050)
        )
        if row["unsafe_commit"] and self.first_commit_s is None:
            self.first_commit_s = row["sim_time_s"]
            self.first_commit_truth = row
        # Parent last/first truth references point at this same returned row;
        # samples are counted once and original HARM priority is unchanged.
        return row
