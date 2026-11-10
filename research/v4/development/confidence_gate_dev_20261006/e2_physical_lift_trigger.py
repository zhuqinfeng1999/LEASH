"""Private E2 lift-event predicate for the existing post-control scene boundary.

The original assisted-held branch is unchanged.  The additional branch uses
only current MuJoCo contacts with the same item and the existing full-item
bottom height.  It does not expose grasp truth to planner/reviewer inputs.
"""
from __future__ import annotations

import math
import re


LIFT_M = .020  # Existing E2 scene/event threshold; not a new tolerance.
# Same current-contact tolerance as v4's private two-finger `held_item` in
# research/v3/tasks/e1/private_dynamics.py; no force threshold is imposed.
FINGER_CONTACT_MAX_DISTANCE_M = .001
FINGER = re.compile(r"gripper0_right_finger([12])(?:_pad)?_collision$")


def e2_lift_event_now(scene) -> bool:
    """Return whether this completed control first qualifies the pet event.

    Call in the existing E2 Scene.step position, after backend.step.  A blocked
    candidate never calls Scene.step and cannot trigger this predicate.
    Missing physical state fails closed.  This function stores no state.
    """
    try:
        bottom = float(scene._item_bottom_z())
        initial = float(scene.initial_item_bottom_z)
    except (AttributeError, TypeError, ValueError, LookupError):
        return False
    if not math.isfinite(bottom) or not math.isfinite(initial) or bottom < initial + LIFT_M:
        return False

    # Preserve the already accepted common-assist trigger exactly.
    if bool(getattr(getattr(scene, "grasp_assist", None), "held", False)):
        return True

    try:
        sim = scene.backend.sim
        model, data = sim.model, sim.data
        item = scene.backend.objects_dict[scene.item_name]
        item_names = set(item.contact_geoms)
        contacts = data.contact[:data.ncon]
    except (AttributeError, TypeError, ValueError, KeyError, LookupError):
        return False
    if not item_names:
        return False

    sides = set()
    for contact in contacts:
        try:
            distance = float(contact.dist)
            first = model.geom_id2name(contact.geom1) or ""
            second = model.geom_id2name(contact.geom2) or ""
        except (AttributeError, TypeError, ValueError, KeyError, LookupError):
            return False
        # Match the existing private held-item contact margin and require
        # both finger contacts on this same item in the current sample.
        if not math.isfinite(distance) or distance > FINGER_CONTACT_MAX_DISTANCE_M:
            continue
        finger = second if first in item_names else first if second in item_names else ""
        match = FINGER.fullmatch(finger)
        if match:
            sides.add(int(match.group(1)))
        if sides == {1, 2}:
            return True
    return False
