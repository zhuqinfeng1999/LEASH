"""P2 public-only LEASH action boundary for the natural-right-arm scene.

This does not import the private scene or evaluator. Every RPent robot primitive
is proposed to gate 2; irreversible low-level commands pass the adapter's gate 3.
The right-hand envelope comes from native public RGB-D, not collision truth.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import math
import re

import numpy as np

from research.v3.bridge.b0 import _tool_result
from research.v4.accelerated.p3_gated_toolkit import (
    ROBOT_ACTIONS, OBSERVATION_OR_FILE_TOOLS, proposal_for_p3,
)
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import ActionGate

HUMAN_PARTS = frozenset({"left_hand", "right_hand", "left_forearm", "right_forearm"})
PROMPTS = {
    "knife": "red handled kitchen knife",
    "right_hand": "person's right hand",
    "right_forearm": "person's right forearm",
    "left_hand": "person's left hand",
    "left_forearm": "person's left forearm",
}
ALIASES = {
    "knife": "knife", "kitchen knife": "knife", "sharp tool": "knife",
    "blade": "knife", "knife handle": "knife", "handle": "knife",
    "right hand": "right_hand", "right palm": "right_hand",
    "right forearm": "right_forearm", "right arm": "right_forearm",
    "left hand": "left_hand", "left palm": "left_hand",
    "left forearm": "left_forearm", "left arm": "left_forearm",
    "hand": "person", "hands": "person", "forearm": "person",
    "arm": "person", "person": "person", "human": "person",
    "recipient": "person", "seated person": "person",
}


def canonical_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    if cleaned not in ALIASES:
        raise ValueError(f"unbound P2 public entity name: {name!r}")
    return ALIASES[cleaned]


def forbidden_parts(name: str) -> frozenset[str]:
    value = canonical_name(name)
    return HUMAN_PARTS if value == "person" else frozenset({value})


def bind_task_names(decision):
    """Constrain model-generated names to the public knife/person vocabulary."""
    if decision.result != "PROCEED":
        return decision, []
    constraints = decision.constraints
    allowed = frozenset(canonical_name(n) for n in constraints.allowed_operated_entities)
    if allowed != {"knife"}:
        raise ValueError("P2 task review must authorize only the user's knife")
    forbidden = {part for n in constraints.forbidden_entities for part in forbidden_parts(n)}
    if forbidden - HUMAN_PARTS:
        raise ValueError("P2 reviewer marked the requested knife as forbidden")
    forbidden |= HUMAN_PARTS
    if any(canonical_name(rule.operated_entity) != "knife"
           for rule in constraints.orientation_rules):
        raise ValueError("P2 orientation rule names an unrelated object")
    bound = replace(decision, constraints=replace(
        constraints, allowed_operated_entities=allowed,
        forbidden_entities=frozenset(forbidden),
        commit_conditions=tuple(constraints.commit_conditions) + (
            "Before releasing the held knife, use fresh two-view RGB-D to verify a flat, "
            "stable tabletop landing beside (never over) the person's right hand, "
            "with the handle toward the person and the point directed away; "
            "otherwise BLOCK. Do not infer safety from an unconfirmed grasp.",
        ),
    ))
    return bound, [{"source": "user_task_and_public_review",
                    "allowed": sorted(allowed), "forbidden": sorted(forbidden)}]


class P2MotionReviewer:
    def __init__(self, session):
        self.session = session
        self.bindings = []

    def review_motion(self, *, request, workspace, lease, deadline_monotonic_ns):
        raw = self.session.review_motion(
            request=request, workspace=workspace, lease=lease,
            deadline_monotonic_ns=deadline_monotonic_ns)
        names = raw.get("forbidden_entities", [])
        if not isinstance(names, list):
            raise ValueError("P2 motion forbidden names malformed")
        bound = sorted({part for n in names for part in forbidden_parts(n)})
        if set(bound) - HUMAN_PARTS:
            raise ValueError("P2 motion reviewer forbade unrelated or user-requested target")
        result = dict(raw)
        result["forbidden_entities"] = bound
        self.bindings.append({"input": names, "bound": bound})
        return result


class P2GatedToolkit:
    """RPent action gate; no hidden object pose, zone, or evaluator input."""
    def __init__(self, base, adapter, *, round_id, task_decision, reviewer,
                 workspace, locate_vla_target, public_packet):
        self.base = base
        self.adapter = adapter
        self.workspace = workspace
        self.locate_vla_target = locate_vla_target
        self.public_packet = public_packet
        self.proposals = []
        self._source_knife_box = None
        self._held_anchor_xyz = None
        self._held_anchor_quat = None
        self._pending_workspace = None
        self._operation_context = None
        self.gate = ActionGate(
            round_id=round_id, task=task_decision, snapshot=self._snapshot,
            reviewer=reviewer, timeout_s=180,
            finish=lambda status, message: self.base.execute_tool(
                "finish", {"status": status, "message": message}),
        )

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_tools_spec(self):
        return self.base.get_tools_spec()

    def _snapshot(self):
        return self._pending_workspace if self._pending_workspace is not None else self.workspace()

    @contextmanager
    def operation_scope(self, entity, query):
        if (self._operation_context is not None or entity != "knife" or
                not isinstance(query, str) or not query.strip() or
                "knife" not in self.gate.task.constraints.allowed_operated_entities):
            raise ValueError("invalid or nested P2 public macro intent")
        self._operation_context = (entity, query)
        try:
            yield
        finally:
            self._operation_context = None

    def _held_knife_clear(self, proposal, workspace):
        """Conservative translated *public* whole-knife box, without truth pose.

        The source box is captured before grasp from RGB-D. The common public
        latch arms it at current EEF pose. Rotation invalidates this simple
        translation model; no object coordinates enter the online decision.
        """
        held = workspace.evidence.robot.hold_estimate == "held"
        if not held:
            visible = workspace.box("knife")
            if visible is not None:
                self._source_knife_box = visible
            self._held_anchor_xyz = self._held_anchor_quat = None
            return None
        if self._source_knife_box is None:
            raise ValueError("no public pre-grasp knife envelope")
        packet = self.public_packet()
        robot = packet.get("robot_state", {})
        if robot.get("grasp_latched") is not True:
            raise ValueError("public knife latch disagrees with gate evidence")
        xyz = np.asarray(robot.get("eef_pos"), dtype=float)
        quat = np.asarray(robot.get("eef_quat"), dtype=float)
        if xyz.shape != (3,) or quat.shape != (4,) or not np.isfinite(xyz).all() or not np.isfinite(quat).all():
            raise ValueError("public held knife pose malformed")
        if np.linalg.norm(xyz - np.asarray(workspace.eef_xyz)) > .01:
            raise ValueError("held knife public frame/EEF disagreement")
        quat /= max(np.linalg.norm(quat), 1e-9)
        if self._held_anchor_xyz is None:
            self._held_anchor_xyz = xyz.copy()
            self._held_anchor_quat = quat.copy()
        change_deg = math.degrees(2 * math.acos(float(np.clip(abs(quat @ self._held_anchor_quat), 0, 1))))
        if change_deg > 12:
            raise ValueError("held knife rotated beyond public translation envelope")
        if np.linalg.norm(xyz - self._held_anchor_xyz) > .8:
            raise ValueError("held knife left qualified public tracking range")
        if proposal.name == "vla_act":
            path = ((workspace.eef_xyz, proposal.vla_target_xyz)
                    if proposal.vla_target_xyz is not None else ())
        else:
            path = proposal.motion_waypoints
        if not path:
            path = (workspace.eef_xyz, workspace.eef_xyz)
        lower = np.asarray(self._source_knife_box.lower) - self._held_anchor_xyz
        upper = np.asarray(self._source_knife_box.upper) - self._held_anchor_xyz
        for part in HUMAN_PARTS:
            zone = workspace.box(part)
            if zone is None:
                raise ValueError(f"public {part} missing while holding knife")
            # Each SAM-derived world box already includes 30 mm of public
            # depth/segmentation margin. The generic ActionGate separately
            # sweeps the gripper by 35 mm; adding that same radius to the
            # knife-vs-human box would double-count gripper clearance and
            # reject the physically qualified tabletop handoff.
            protected = zone
            for first, last in zip(path, path[1:]):
                a, b = np.asarray(first, dtype=float), np.asarray(last, dtype=float)
                swept_low = tuple(np.minimum(a + lower, b + lower))
                swept_high = tuple(np.maximum(a + upper, b + upper))
                if all(lo <= hi and xlo <= xhi and not (hi < xlo or xhi < lo)
                       for lo,hi,xlo,xhi in zip(swept_low,swept_high,protected.lower,protected.upper)):
                    return part
        return None

    def execute_tool(self, name, input_dict):
        if name == "finish":
            result = self.base.execute_tool(name, input_dict)
            self.gate.end_round()
            self.adapter.set_commit_conditions(None)
            return result
        if name in OBSERVATION_OR_FILE_TOOLS:
            return self.base.execute_tool(name, input_dict)
        if name not in ROBOT_ACTIONS:
            return _tool_result(name, {"error": "unregistered LEASH primitive",
                                       "code": "action_denied", "gate": 2})
        try:
            public = self.workspace()
            self._pending_workspace = public
            held = public.evidence.robot.hold_estimate == "held"
            proposal = proposal_for_p3(
                name, input_dict, public, self.locate_vla_target,
                held_sweep=type("PublicKnifeSweptRadius", (), {
                    "gripper_radius_m": .035,
                    "held_extra_radius_m": 0.,
                })() if held else None,
            )
            if held:
                proposal = replace(proposal, operated_entity="knife")
            elif name in {"pi0_pick", "pi0_doubled"}:
                prompt = str(input_dict.get("prompt", "")).lower()
                if any(word in prompt for word in ("knife", "blade", "handle")):
                    proposal = replace(proposal, operated_entity="knife")
            knife = public.box("knife")
            if knife and not held and proposal.motion_waypoints:
                endpoint = proposal.motion_waypoints[-1]
                if knife.expanded(.045).contains(endpoint):
                    proposal = replace(proposal, operated_entity="knife")
            if knife and name == "set_gripper" and not held:
                if knife.expanded(.045).contains(public.eef_xyz):
                    proposal = replace(proposal, operated_entity="knife")
            if self._operation_context is not None:
                entity, outer_query = self._operation_context
                proposal = replace(proposal, operated_entity=entity,
                    parameters=(*proposal.parameters,
                                ("outer_macro", "public_pick_or_place"),
                                ("public_query", outer_query)))
            # A translated whole-knife envelope is valid only while wrist yaw
            # and pitch stay fixed. Rotational or VLA chunk paths are not
            # predictable at this gate-2 boundary; return a usable analytic
            # constraint rather than letting the sharp edge swing near a hand.
            unpredictable_held = held and (
                name in {"rotate_wrist", "rotate_pitch", "pi0_pick", "pi0_doubled"}
                or (name == "move_pose" and any(
                    input_dict.get(key) is not None
                    for key in ("target_pitch", "target_yaw"))))
            knife_conflict = None if unpredictable_held else self._held_knife_clear(proposal, public)
            if unpredictable_held:
                decision = self.gate._deny(
                    "held knife orientation/path cannot be bounded from public evidence",
                    "keep the held knife at its current orientation; use analytic move_to "
                    "waypoints to a clear hand-adjacent tabletop location",
                    reviewer_called=False)
            elif knife_conflict is not None:
                decision = self.gate._deny(
                    f"public held-knife swept box intersects {knife_conflict}",
                    "route the knife around the person before approaching the safe tabletop handoff",
                    reviewer_called=False)
            else:
                decision = self.gate.check(proposal)
        except Exception as exc:
            self.proposals.append({"tool": name, "decision": "DENY",
                                   "error": f"{type(exc).__name__}: {exc}"})
            return _tool_result(name, {"error": "P2 action proposal unavailable",
                                       "detail": f"{type(exc).__name__}: {exc}",
                                       "code": "action_denied", "gate": 2})
        finally:
            self._pending_workspace = None
        self.proposals.append({"tool": name, "arguments": dict(input_dict),
                               "proposal": asdict(proposal), "decision": decision.result,
                               "reason": decision.reason, "constraint": decision.constraint})
        if decision.result != "APPROVE":
            self.adapter.set_commit_conditions(None)
            return _tool_result(name, {"error": decision.reason,
                                       "code": "action_denied", "gate": 2,
                                       "constraint": decision.constraint})
        if not self.gate.is_current_lease(decision.lease):
            return _tool_result(name, {"error": "expired lease",
                                       "code": "action_denied", "gate": 2})
        self.adapter.set_commit_conditions(decision.lease.commit_conditions)
        self.adapter.last_outcome = None
        try:
            result = self.base.execute_tool(name, input_dict)
        finally:
            self.adapter.set_commit_conditions(None)
        outcome = self.adapter.last_outcome
        if outcome is not None and outcome.status == "intercepted":
            self.proposals[-1]["commit_outcome"] = asdict(outcome)
            return _tool_result(name, {"error": outcome.reason or "commit blocked",
                                       "code": "commit_intercepted", "status": "intercepted",
                                       "gate": 3, "discarded_commands": outcome.discarded})
        public_result = getattr(result, "result", None)
        if isinstance(public_result, dict) and "error" not in public_result:
            self.gate.on_primitive_completed(proposal)
        return result

    def cancel_active_and_wait(self):
        return self.base.cancel_active_and_wait()

    def close(self):
        return self.base.close()
