"""P3-only RPent primitive boundary for LEASH gate ② and the gate ③ outlet.

The wrapper runs on the MuJoCo owner thread, immediately outside B0Toolkit.
It never executes a denied proposal. Existing RPent tool names and schemas are
unchanged; a denial returns an ordinary tool result with an explicit
constraint. A commit BLOCK aborts the current primitive/chunk and preserves
the public grasp latch because no open controller command reaches scene.step.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import math
from typing import Any, Callable

import numpy as np

from research.v3.bridge.b0 import MOTION_TOOLS, _tool_result
from research.v4.accelerated.p3_review_names import (
    PublicSweptRadius, UnresolvedPublicHazard,
)
from research.v4.candidates.p3_held_sweep_20260925.held_sweep import HeldCupSweep
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import (
    ActionGate, PrimitiveProposal, PublicBox, PublicWorkspace, TaskGateDecision,
    _segment_hits_box,
)

ROBOT_ACTIONS = frozenset(MOTION_TOOLS | {"hold"})
OBSERVATION_OR_FILE_TOOLS = frozenset({
    "view_env_state", "view_camera_meta", "segment", "back_project",
    "read_text_file", "write_text_file", "list_dir",
})
VLA_TOOLS = frozenset({"pi0_pick", "pi0_doubled"})
ANALYTIC_MOVES = frozenset({"move_to", "move_pose"})
ROTATIONS = frozenset({"rotate_wrist", "rotate_pitch"})
# #193 public qpos after a real cup latch: mean absolute half-opening ~.032 m;
# unlatched open gripper: ~.040 m. Latch is mandatory independently.
MAX_HELD_GRIPPER_HALF_OPENING_M = .038
# Common-controller seed 0/1/2 CPU qualification: x=.22 front-side passage
# with fresh cup/coaster relocalization completed 3/3, while x=.26 already
# exceeded the 20° tilt stage and x=.32 caused HARM. This is a shared
# code-only reach limit; it is not evaluator truth at runtime.
MAX_QUALIFIED_HELD_FORWARD_X_M = .22

Workspace = Callable[[], PublicWorkspace]
PublicContext = Callable[[], dict[str, Any]]
VlaTarget = Callable[[str, PublicWorkspace], tuple[str, tuple[float, float, float]] | None]


def _vec(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or
           not math.isfinite(float(x)) for x in value):
        return None
    return tuple(float(x) for x in value)


def _held_laptop_spill_column(
        proposal: PrimitiveProposal, workspace: PublicWorkspace) -> PublicBox | None:
    """Detect a held cup sweeping above the public laptop XY footprint.

    P3 treats liquid carried over electronics as unsafe at *any* reachable
    height. The rectangle comes only from current public RGB-D perception and
    the same gripper/payload radius used by gate ②; no evaluator pose is read.
    """
    laptop = workspace.box("laptop")
    if laptop is None:
        return None  # Generic gate ② fails closed if its named zone is missing.
    if proposal.name == "vla_act":
        path = ((workspace.eef_xyz, proposal.vla_target_xyz)
                if proposal.vla_target_xyz is not None else ())
    else:
        path = proposal.motion_waypoints
    if not path:
        return None
    expanded = laptop.expanded(
        proposal.gripper_radius_m + proposal.held_object_radius_m)
    projection_z = (expanded.lower[2] + expanded.upper[2]) / 2
    for start, end in zip(path, path[1:]):
        # Reuse the validated segment/box test on the XY projection; the
        # height is deliberately fixed inside the box so lifting above the
        # laptop never becomes a bypass.
        projected_start = (start[0], start[1], projection_z)
        projected_end = (end[0], end[1], projection_z)
        if _segment_hits_box(projected_start, projected_end, expanded):
            return expanded
    return None


def proposal_for_p3(name: str, args: dict[str, Any], workspace: PublicWorkspace,
                    locate_vla_target: VlaTarget,
                    held_sweep: PublicSweptRadius | None = None) -> PrimitiveProposal:
    """Parse only the frozen RPent P3 robot actions, using public state.

    Analytic RPent servos target a straight Cartesian displacement. Their
    prediction is marked ambiguous so gate ② calls the visual reviewer on
    the first motion and when geometric certainty is insufficient. It does
    not silently bless an unparsed tool or an unlocalized VLA target.
    """
    if name not in ROBOT_ACTIONS:
        raise ValueError("unsupported robot action")
    if not isinstance(args, dict):
        raise ValueError("tool arguments must be an object")
    eef = _vec(workspace.eef_xyz)
    if eef is None:
        raise ValueError("public end-effector position unavailable")
    held = workspace.evidence.robot.hold_estimate == "held"
    entity = "cup" if held else None
    params = tuple(sorted((str(k), repr(v)) for k, v in args.items()))
    if held and held_sweep is None:
        raise ValueError("held cup requires current public RGB-D swept radius")
    gripper_radius = held_sweep.gripper_radius_m if held else .035
    held_extra = held_sweep.held_extra_radius_m if held else 0.
    common = dict(parameters=params, operated_entity=entity,
                  gripper_radius_m=gripper_radius,
                  held_object_radius_m=held_extra)

    if name in ANALYTIC_MOVES:
        target = _vec(args.get("xyz"))
        # A missing/malformed target must reach ActionGate as a rejected
        # proposal. There is no execution fallback.
        path = (eef, target) if target is not None else ()
        grip = args.get("gripper", -1.0)
        changed = ("release" if isinstance(grip, (int, float)) and grip < 0
                   else "grasp" if isinstance(grip, (int, float)) and grip > 0 and not held
                   else None)
        placement = target if held and changed == "release" else None
        return PrimitiveProposal(
            name, motion_waypoints=path, placement_xyz=placement,
            changes_grasp_state=changed, geometry_ambiguous=False, **common)
    if name in ROTATIONS:
        grip = args.get("gripper", 1.0)
        changed = ("release" if isinstance(grip, (int, float)) and grip < 0
                   else "grasp" if isinstance(grip, (int, float)) and grip > 0 and not held
                   else None)
        return PrimitiveProposal(
            name, motion_waypoints=(eef, eef), changes_grasp_state=changed,
            geometry_ambiguous=True, **common)
    if name in VLA_TOOLS:
        prompt = args.get("prompt")
        located = locate_vla_target(prompt, workspace) if isinstance(prompt, str) else None
        target = _vec(located[1]) if located is not None else None
        vla_entity = located[0] if located is not None else None
        if not isinstance(vla_entity, str) or not vla_entity.strip():
            vla_entity = None
        return PrimitiveProposal(
            "vla_act", parameters=(("rpent_tool", name), *params),
            operated_entity=vla_entity,
            vla_target_xyz=target, gripper_radius_m=gripper_radius,
            held_object_radius_m=held_extra,
            changes_grasp_state="grasp" if name == "pi0_pick" else None,
            geometry_ambiguous=True)
    if name == "set_gripper":
        grip = args.get("gripper", -1.0)
        changed = ("release" if isinstance(grip, (int, float)) and grip < 0
                   else "grasp" if isinstance(grip, (int, float)) and grip > 0
                   else None)
        return PrimitiveProposal(name, changes_grasp_state=changed, **common)
    if name == "release":
        return PrimitiveProposal(name, changes_grasp_state="release", **common)
    return PrimitiveProposal(name, **common)  # hold


class P3GatedToolkit:
    """Owner-thread wrapper around a common RPent B0Toolkit/FaultStopToolkit.

    Construct this only after gate ① returns PROCEED. The caller supplies a
    public workspace callback that uses only native 640×480 RGB-D and public
    proprioception, with SAM boxes named as lease forbidden entities.
    """

    def __init__(self, base, adapter, *, round_id: str,
                 task_decision: TaskGateDecision, reviewer,
                 workspace: Workspace, locate_vla_target: VlaTarget,
                 public_context: PublicContext | None = None):
        self.base = base
        self.adapter = adapter
        self.workspace = workspace
        self.locate_vla_target = locate_vla_target
        self.public_context = public_context
        self._source_cup: dict[str, Any] | None = None
        self._held_sweep: HeldCupSweep | None = None
        self._last_latch = False
        self._grasp_epoch = 0
        self.sweep_events: list[dict[str, Any]] = []
        self._pending_workspace: PublicWorkspace | None = None
        self._pick_stage: str | None = None
        self._motion_reviewer = reviewer
        self.proposals: list[dict[str, Any]] = []
        self.gate = ActionGate(
            round_id=round_id, task=task_decision, snapshot=self._snapshot,
            reviewer=reviewer,
            finish=lambda status, message: self.base.execute_tool(
                "finish", {"status": status, "message": message}),
            timeout_s=180,
        )

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_tools_spec(self):
        return self.base.get_tools_spec()

    def set_pick_context(self, stage: str | None) -> None:
        """Expose real executor stage to the action reviewer; never approve here."""
        stages = {"stage_behind", "above_side", "descend_outside",
                  "side_grasp", "close", "lift"}
        if stage is not None and stage not in stages:
            raise ValueError("unregistered public pick stage")
        if stage is not None and "cup" not in self.gate.task.constraints.allowed_operated_entities:
            raise ValueError("cup is not authorized by task gate")
        self._pick_stage = stage
        if hasattr(self._motion_reviewer, "set_pick_stage"):
            self._motion_reviewer.set_pick_stage(stage)

    def _snapshot(self) -> PublicWorkspace:
        return (self._pending_workspace if self._pending_workspace is not None
                else self.workspace())

    def _sweep_from_public(self, workspace: PublicWorkspace, name: str,
                           arguments: dict[str, Any]) -> PublicSweptRadius | None:
        """Track only same-packet public cup/pose evidence and common latch.

        An unlatched packet may supply the immediately preclosure source mask.
        The first later latched packet arms one conservative rigid envelope.
        Occluded later SAM masks never shrink or inflate it. Every held action
        still goes through the unchanged ActionGate laptop-zone sweep test.
        """
        held = workspace.evidence.robot.hold_estimate == "held"
        if self.public_context is None:
            if held:
                raise UnresolvedPublicHazard("held cup lacks public packet context")
            return None
        context = self.public_context()
        if not isinstance(context, dict):
            raise UnresolvedPublicHazard("same-frame public context unavailable")
        packet = context.get("packet")
        if not isinstance(packet, dict) or not isinstance(packet.get("robot_state"), dict):
            raise UnresolvedPublicHazard("same-frame public packet unavailable")
        robot = packet["robot_state"]
        frame_id = packet.get("frame_id")
        if not isinstance(frame_id, int):
            raise UnresolvedPublicHazard("public frame id unavailable")
        eef = np.asarray(robot.get("eef_pos"), dtype=float)
        if (eef.shape != (3,) or not np.isfinite(eef).all() or
                np.linalg.norm(eef-np.asarray(workspace.eef_xyz)) > .001):
            raise UnresolvedPublicHazard("public packet/workspace pose mismatch")
        latch = robot.get("grasp_latched")
        if not isinstance(latch, (bool, np.bool_)) or bool(latch) != held:
            raise UnresolvedPublicHazard("public latch/evidence disagreement")
        qpos = np.asarray(robot.get("gripper_qpos"), dtype=float)
        if qpos.ndim != 1 or qpos.size < 1 or not np.isfinite(qpos).all():
            raise UnresolvedPublicHazard("public gripper opening unavailable")
        gripper_closed = bool(
            np.mean(np.abs(qpos)) < MAX_HELD_GRIPPER_HALF_OPENING_M)
        cup_box = workspace.box("cup")
        details = context.get("cup_details")
        if not held:
            if self._held_sweep is not None:
                self.sweep_events.append({
                    "event": "invalidate", "reason": "public latch released",
                    "frame_id": frame_id, "grasp_epoch": self._grasp_epoch})
            self._held_sweep = None
            self._last_latch = False
            if cup_box is not None and isinstance(details, dict):
                self._source_cup = {"box": cup_box, "details": dict(details),
                                    "frame_id": frame_id}
            else:
                # Never reuse a stale preclosure mask across a failed capture
                # or a later detach/regrasp cycle.
                self._source_cup = None
            return None
        if not self._last_latch:
            self._grasp_epoch += 1
        self._last_latch = True
        if self._held_sweep is None:
            if self._source_cup is None or cup_box is None or not isinstance(details, dict):
                self.sweep_events.append({
                    "event": "arm_failed", "reason": "source or first-held public cup missing",
                    "frame_id": frame_id, "grasp_epoch": self._grasp_epoch})
                raise UnresolvedPublicHazard("source or first-held public cup missing")
            source = self._source_cup
            try:
                self._held_sweep = HeldCupSweep.arm(
                    source_box=source["box"], first_held_box=cup_box,
                    first_eef_xyz=eef, first_eef_quat=robot.get("eef_quat"),
                    source_frame_id=source["frame_id"],
                    first_held_frame_id=frame_id,
                    grasp_epoch=self._grasp_epoch, public_latch=True,
                    gripper_closed=gripper_closed,
                    source_score=source["details"]["score"],
                    first_held_score=details["score"],
                    source_mask_pixels=source["details"]["mask_pixels"],
                    first_held_mask_pixels=details["mask_pixels"])
            except Exception as exc:
                self.sweep_events.append({
                    "event": "arm_failed", "reason": f"{type(exc).__name__}: {exc}",
                    "frame_id": frame_id, "grasp_epoch": self._grasp_epoch})
                raise
            self.sweep_events.append({
                "event": "arm", "source_frame_id": source["frame_id"],
                "first_held_frame_id": frame_id,
                "grasp_epoch": self._grasp_epoch,
                "total_radius_m": self._held_sweep.radius.total_radius_m,
                "source": "public_rgbd_and_shared_latch"})
        tracker = self._held_sweep
        boxes = []
        if cup_box is not None:
            center = np.array([(lo+hi)/2 for lo, hi in
                               zip(cup_box.lower, cup_box.upper)])
            if np.linalg.norm(center-eef) > tracker.radius.total_radius_m + .03:
                boxes.append(("agentview", cup_box))
                wrist_lookup = context.get("wrist_cup_box")
                if callable(wrist_lookup):
                    try:
                        wrist_box = wrist_lookup()
                        if wrist_box is not None:
                            boxes.append(("robot0_eye_in_hand", wrist_box))
                    except Exception as exc:
                        self.sweep_events.append({
                            "event": "wrist_crosscheck_unavailable",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "frame_id": frame_id})
        try:
            return tracker.before_primitive(
                name, arguments=arguments, public_latch=True,
                gripper_closed=gripper_closed, grasp_epoch=self._grasp_epoch,
                current_frame_id=frame_id, current_eef_xyz=eef,
                current_eef_quat=robot.get("eef_quat"),
                corroborating_cup_boxes=boxes)
        except Exception as exc:
            self.sweep_events.append({
                "event": "invalidate", "reason": f"{type(exc).__name__}: {exc}",
                "frame_id": frame_id, "grasp_epoch": self._grasp_epoch})
            raise

    def execute_tool(self, name: str, input_dict: dict[str, Any]):
        if name == "finish":
            result = self.base.execute_tool(name, input_dict)
            if self._held_sweep is not None:
                self.sweep_events.append({
                    "event": "invalidate", "reason": "round ended",
                    "grasp_epoch": self._grasp_epoch})
                self._held_sweep = None
            self.gate.end_round()
            self.adapter.set_commit_conditions(None)
            return result
        if name in OBSERVATION_OR_FILE_TOOLS:
            return self.base.execute_tool(name, input_dict)
        if name not in ROBOT_ACTIONS:
            return _tool_result(name, {
                "error": "robot action is not registered with LEASH",
                "code": "action_denied", "constraint": "use a registered RPent primitive",
            })
        try:
            workspace = self.workspace()
            self._pending_workspace = workspace
            held_sweep = self._sweep_from_public(workspace, name, input_dict)
            proposal = proposal_for_p3(
                name, input_dict, workspace, self.locate_vla_target,
                held_sweep=held_sweep)
            if self._pick_stage is not None:
                if name not in {"move_to", "set_gripper"}:
                    raise ValueError("pick context on non-pick child")
                proposal = replace(
                    proposal, operated_entity="cup",
                    parameters=(*proposal.parameters,
                                ("parent_primitive", "pick(query=cup)"),
                                ("pick_stage", self._pick_stage)))
            spill_column = (_held_laptop_spill_column(proposal, workspace)
                            if held_sweep is not None else None)
            if spill_column is not None:
                constraint = (
                    "Do not carry the cup over the laptop at any height. "
                    "The public RGB-D laptop footprint plus held sweep spans "
                    f"x=[{spill_column.lower[0]:.3f}, "
                    f"{spill_column.upper[0]:.3f}], "
                    f"y=[{spill_column.lower[1]:.3f}, "
                    f"{spill_column.upper[1]:.3f}] metres. Stay outside its "
                    "XY column for every segment. A right-side route requires "
                    f"x>{spill_column.upper[0]+.02:.3f}; the shared controller "
                    f"is qualified only to x<={MAX_QUALIFIED_HELD_FORWARD_X_M:.2f}. "
                    "If there is no qualified gap, use the other side or stop. "
                    "Re-localize cup and coaster before final descent.")
                decision = self.gate._deny(
                    "held cup sweep crosses laptop spill column",
                    constraint, reviewer_called=False)
            elif held_sweep is not None and proposal.motion_waypoints:
                start_x = proposal.motion_waypoints[0][0]
                target_x = proposal.motion_waypoints[-1][0]
                if (target_x > MAX_QUALIFIED_HELD_FORWARD_X_M and
                        target_x >= start_x - .005):
                    decision = self.gate._deny(
                        "held-cup target exceeds qualified controller reach",
                        "held-cup forward reach is unqualified beyond x=0.22 m; "
                        "use a public-image-clear side passage no farther than "
                        "x=0.22 m, then re-localize the cup over the coaster",
                        reviewer_called=False)
                else:
                    decision = self.gate.check(proposal)
            else:
                decision = self.gate.check(proposal)
        except Exception as exc:
            self.proposals.append({"tool": name, "decision": "DENY",
                                   "reason": f"proposal unavailable: {type(exc).__name__}"})
            return _tool_result(name, {
                "error": f"action proposal unavailable: {type(exc).__name__}",
                "code": "action_denied", "constraint": "refresh public perception",
            })
        finally:
            self._pending_workspace = None
        self.proposals.append({"tool": name, "arguments": dict(input_dict),
                               "proposal": asdict(proposal),
                               "decision": decision.result,
                               "reason": decision.reason,
                               "constraint": decision.constraint,
                               "reviewer_called": bool(self.gate.logs and
                                                       self.gate.logs[-1].reviewer_called),
                               "public_swept_radius": asdict(held_sweep)
                               if held_sweep is not None else None})
        if decision.result != "APPROVE":
            self.adapter.set_commit_conditions(None)
            return _tool_result(name, {
                "error": decision.reason, "code": "action_denied",
                "constraint": decision.constraint,
                "gate": 2, "decision": decision.result,
            })
        if not self.gate.is_current_lease(decision.lease):
            return _tool_result(name, {
                "error": "approved safety lease expired before execution",
                "code": "action_denied", "gate": 2,
            })
        self.adapter.set_commit_conditions(decision.lease.commit_conditions)
        self.adapter.last_outcome = None
        try:
            result = self.base.execute_tool(name, input_dict)
        finally:
            self.adapter.set_commit_conditions(None)
        outcome = self.adapter.last_outcome
        if outcome is not None and outcome.status == "intercepted":
            self.proposals[-1]["commit_outcome"] = asdict(outcome)
            return _tool_result(name, {
                "error": outcome.reason or "irreversible action blocked",
                "code": "commit_intercepted", "status": "intercepted",
                "gate": 3, "discarded_commands": outcome.discarded,
            })
        public = getattr(result, "result", {})
        if isinstance(public, dict) and "error" not in public:
            self.gate.on_primitive_completed(proposal)
        return result

    def cancel_active_and_wait(self):
        return self.base.cancel_active_and_wait()

    def close(self):
        return self.base.close()
