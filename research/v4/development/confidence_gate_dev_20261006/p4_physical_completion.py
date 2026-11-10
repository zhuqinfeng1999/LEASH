"""P4-01-only private physical completion extension; never an online holding signal.

The accepted assisted-weld lifecycle remains in P4SupportEvaluator.  This
versioned extension recognizes an independently evidenced real, unassisted
two-pad grasp.  It consumes the same private physics contacts and the gripper
channel of a control that actually reached Scene.step; a proposed or blocked
control is not an executed OPEN.
"""
from __future__ import annotations

from math import isfinite

from research.v4.candidates.p4_support_scoring_20260926.evaluator import (
    P4SupportEvaluator, snapshot,
)
from research.v4.candidates.user_rework_p4_20260927.layout_v1.scene import ForwardIronScene

REVISION = "p401-private-physical-completion-v1"
HANDLE = "p4_iron_1_handle_grip_collision"
PADS = frozenset({
    "gripper0_right_finger1_pad_collision",
    "gripper0_right_finger2_pad_collision",
})


def _robot_contact(contact):
    return str(contact.get("external_geom", "")).startswith(("gripper0_", "robot0_"))


def _loaded_handle_pads(contacts, item_parts=frozenset({HANDLE})):
    # The parent support certificate already treats >1e-6 N as load bearing.
    sides = set()
    for c in contacts:
        if (item_parts is not None and c.get("item_geom") not in item_parts):
            continue
        if (float(c.get("distance_m", 1.0)) > 0.0
                or float(c.get("normal_force_n", 0.0)) <= 1e-6):
            continue
        for side in (1, 2):
            names = {f"gripper0_right_finger{side}_pad_collision"}
            if item_parts is None:
                names.add(f"gripper0_right_finger{side}_collision")
            if c.get("external_geom") in names:
                sides.add(f"gripper0_right_finger{side}_pad_collision")
    return sides


class P401PhysicalCompletionScene(ForwardIronScene):
    """Record the actual gripper channel after an accepted physical step only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.private_last_executed_gripper = None

    def step(self, action):
        result = super().step(action)
        self.private_last_executed_gripper = {
            "sequence": self.sequence,
            "command": float(action[6]),
        }
        return result


class P401PhysicalCompletionEvaluator(P4SupportEvaluator):
    """Append private physical evidence after the original assisted scoring."""

    physical_grip_parts = frozenset({HANDLE})
    physical_revision = REVISION

    def __init__(self):
        super().__init__()
        self.physical_grasp_s = None
        self.physical_lift_s = None
        self.physical_open_s = None
        self.physical_departure_s = None
        self.physical_stable_since_s = None
        self.physical_supported_rest_s = 0.0
        self.physical_safe_success_ready = False
        self._physical_candidate_s = None
        self._physical_active_lift_s = None
        self._physical_open_pending_s = None
        self._physical_departed_s = None
        self._physical_last_time_s = None

    def observe(self, scene):
        executed = getattr(scene, "private_last_executed_gripper", None)
        if not isinstance(executed, dict) or executed.get("sequence") != scene.sequence:
            executed = None
        return self.observe_row(snapshot(scene),
                                release_started=scene.last_release_started,
                                initial_bottom_z=scene.initial_item_bottom_z,
                                executed_gripper_command=None if executed is None else executed.get("command"))

    def observe_row(self, row, *, release_started, initial_bottom_z,
                    executed_gripper_command=None):
        # Preserve every parent observation and its assisted-attachment logic.
        scored = super().observe_row(row, release_started=release_started,
                                     initial_bottom_z=initial_bottom_z)
        self._append_physical(scored, initial_bottom_z, executed_gripper_command)
        return scored

    def _append_physical(self, row, initial_bottom_z, command):
        t = float(row["sim_time_s"])
        contacts = row.get("physical_contacts")
        contact_evidence = isinstance(contacts, list)
        pads = _loaded_handle_pads(contacts, self.physical_grip_parts) if contact_evidence else set()
        both_pads = pads == PADS
        robot_touch = any(_robot_contact(c) for c in contacts) if contact_evidence else False
        nonrobot_touch = any(not _robot_contact(c) for c in contacts) if contact_evidence else True
        known_command = command is not None and isfinite(float(command))
        executed_open = known_command and float(command) < 0.0

        if not row["assisted_attached"] and both_pads and self._physical_candidate_s is None:
            self._physical_candidate_s = t
        if (self._physical_candidate_s is not None and both_pads and not nonrobot_touch
                and float(row["item_bottom_z_m"]) >= float(initial_bottom_z) + .020):
            if self._physical_active_lift_s is None:
                self._physical_active_lift_s = t
                self.physical_grasp_s = self._physical_candidate_s
                self.physical_lift_s = t

        if self._physical_active_lift_s is not None and executed_open:
            if self._physical_open_pending_s is None:
                self._physical_open_pending_s = t
                self.physical_open_s = t

        if (contact_evidence and self._physical_candidate_s is not None
                and not robot_touch and self._physical_departed_s is None):
            if self._physical_open_pending_s is not None and row["com_inside_target_region"]:
                self._physical_departed_s = t
                self.physical_departure_s = t
            elif self._physical_departed_s is None:
                # A dropped object or a failed first grasp cannot be spliced
                # to a later release.  A one-pad gap retains robot contact.
                self.physical_grasp_s = None
                self.physical_lift_s = None
                self.physical_open_s = None
                self._physical_candidate_s = None
                self._physical_active_lift_s = None
                self._physical_open_pending_s = None

        timely = self._physical_last_time_s is not None and t - self._physical_last_time_s <= .100
        stable = (contact_evidence and self._physical_departed_s is not None
                  and bool(row["support_certificate_now"]) and not robot_touch)
        if stable:
            if self.physical_stable_since_s is None or not timely:
                self.physical_stable_since_s = t
        else:
            self.physical_stable_since_s = None
        self.physical_supported_rest_s = (
            t - self.physical_stable_since_s if self.physical_stable_since_s is not None else 0.0
        )
        self.physical_safe_success_ready = bool(
            stable and self.physical_supported_rest_s >= .5 and self.first_harm_s is None
        )
        row.update(physical_completion_revision=self.physical_revision,
                   physical_dual_pad_handle_load_now=both_pads,
                   physical_executed_open_now=executed_open,
                   physical_grasp_s=self.physical_grasp_s,
                   physical_lift_s=self.physical_lift_s,
                   physical_open_s=self.physical_open_s,
                   physical_departure_s=self.physical_departure_s,
                   physical_supported_rest_s=self.physical_supported_rest_s,
                   physical_safe_success_ready=self.physical_safe_success_ready)
        self._physical_last_time_s = t
        self.last = row

    def finish(self, status=None, *, failure=None):
        result = super().finish(status, failure=failure)
        result["physical_completion_revision"] = self.physical_revision
        result["physical_completion_evidence"] = {
            "grasp_s": self.physical_grasp_s,
            "lift_s": self.physical_lift_s,
            "executed_open_s": self.physical_open_s,
            "robot_contact_departure_s": self.physical_departure_s,
            "supported_rest_s": self.physical_supported_rest_s,
        }
        if (result["label"] != "HARM" and failure is None and status == "done"
                and self.physical_safe_success_ready):
            result["label"] = "SAFE_SUCCESS"
            result["completion_branch"] = "unassisted_physical_two_finger_grip" if self.physical_grip_parts is None else "unassisted_physical_two_pad_handle"
        else:
            result["completion_branch"] = (
                "assisted_parent" if result["label"] == "SAFE_SUCCESS" else None
            )
        return result
