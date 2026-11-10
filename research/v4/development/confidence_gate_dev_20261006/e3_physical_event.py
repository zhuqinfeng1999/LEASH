"""Private E3 physical-lift coverage at its existing executed-control boundary.

This preserves the accepted assist path and all original pour/human predicates.
It adds no planner/reviewer input, controller command, simulation step, or state
machine. Historical E3 results are not reclassified by this module.
"""
from __future__ import annotations

from math import isfinite

from research.v4.accelerated.e3_offline_scoring import (
    E3OfflineEvaluator,
    snapshot as original_snapshot,
)
from .e2_physical_lift_trigger import (
    FINGER,
    FINGER_CONTACT_MAX_DISTANCE_M,
    e2_lift_event_now,
)


REVISION = "e3-physical-held-event-dev1-20261008"
MAX_CONTACT_SAMPLES = 8  # Evidence log bound, never a decision threshold.


class E3PhysicalEventMixin:
    """Supplement the parent's trigger only after a successful real control."""

    def step(self, action):
        result = super().step(action)
        if self.trigger_time_s is None and e2_lift_event_now(self):
            self.trigger_time_s = float(self.backend.sim.data.time)
        return result


def _private_lift_evidence(scene):
    """Log bounded physical evidence without deciding from the log limit."""
    evidence = {
        "physical_item_bottom_z_m": None,
        "physical_initial_item_bottom_z_m": None,
        "physical_assist_held": None,
        "physical_finger_item_contacts": [],
        "physical_finger_item_contact_count": None,
        "physical_contact_log_truncated": False,
        "physical_evidence_status": "unavailable",
    }
    try:
        bottom = float(scene._item_bottom_z())
        initial = float(scene.initial_item_bottom_z)
        assist_held = scene.grasp_assist.held
        if not isfinite(bottom) or not isfinite(initial) or type(assist_held) is not bool:
            return evidence
        evidence["physical_item_bottom_z_m"] = bottom
        evidence["physical_initial_item_bottom_z_m"] = initial
        evidence["physical_assist_held"] = assist_held
        sim = scene.backend.sim
        model, data = sim.model, sim.data
        item = scene.backend.objects_dict[scene.item_name]
        item_geoms = set(item.contact_geoms)
        if not item_geoms:
            return evidence
        count = 0
        for contact in data.contact[:data.ncon]:
            distance = float(contact.dist)
            first = model.geom_id2name(contact.geom1) or ""
            second = model.geom_id2name(contact.geom2) or ""
            finger = second if first in item_geoms else first if second in item_geoms else ""
            match = FINGER.fullmatch(finger)
            if not match:
                continue
            count += 1
            if len(evidence["physical_finger_item_contacts"]) < MAX_CONTACT_SAMPLES:
                evidence["physical_finger_item_contacts"].append({
                    "finger": int(match.group(1)),
                    "item_geom": first if first in item_geoms else second,
                    "contact_distance_m": distance if isfinite(distance) else None,
                    "within_original_contact_tolerance": bool(
                        isfinite(distance)
                        and distance <= FINGER_CONTACT_MAX_DISTANCE_M),
                })
        evidence["physical_finger_item_contact_count"] = count
        evidence["physical_contact_log_truncated"] = count > MAX_CONTACT_SAMPLES
        evidence["physical_evidence_status"] = "complete"
    except (AttributeError, TypeError, ValueError, KeyError, LookupError):
        pass
    return evidence


def e3_physical_snapshot(scene):
    """Call the original E3 snapshot once, then restore qualified real lift."""
    sample = original_snapshot(scene)
    evidence = _private_lift_evidence(scene)
    sample.update(evidence)
    qualified = e2_lift_event_now(scene)
    sample["physical_lift_qualified_now"] = qualified
    sample["physical_event_revision"] = REVISION
    if qualified:
        bottom = evidence["physical_item_bottom_z_m"]
        initial = evidence["physical_initial_item_bottom_z_m"]
        if bottom is not None and initial is not None:
            sample["held_lift_m"] = max(float(sample["held_lift_m"]), bottom - initial)
    return sample


class E3PhysicalEventEvaluator(E3OfflineEvaluator):
    """Feed the corrected private sample to the original scorer once."""

    def observe(self, scene):
        sample = e3_physical_snapshot(scene)
        row = super().observe(sample)
        row.update({key: value for key, value in sample.items()
                    if key.startswith("physical_")})
        return row
