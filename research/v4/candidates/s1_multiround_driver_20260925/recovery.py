"""Owner-thread S1 inter-round recovery using the existing gate-2/3 path.

This is an executor recovery primitive, not an extra planner instruction.
The first-round task lease has ended, so a fresh recovery ActionGate checks
current public workspace against the same task constraints before moving.
"""
from __future__ import annotations

from dataclasses import asdict
import numpy as np

from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import (
    ActionGate, PrimitiveProposal,
)
from .bridge import RecoveryRecord
from .park import park_to_public_start


class AdapterControlOutlet:
    is_common_outlet = True

    def __init__(self, adapter, *, action_gate_checked: bool):
        self.adapter = adapter
        self.action_gate_checked = action_gate_checked

    def execute(self, action):
        # For Ours this is P3GatedAdapter.step, the shared gate-3 low-level
        # outlet. For B0 it is R2V42Adapter.step. No direct scene.step.
        self.adapter.step(action)


def recover_between_s1_rounds(*, method: str, scene, adapter,
                              initial_public_eef_xyz,
                              task_decision=None, reviewer=None,
                              workspace=None, round_id: str = "") -> tuple[RecoveryRecord, list[dict]]:
    """Physically park if empty; preserve a held pan with one closed HOLD step.

    The return's `action_gate_checked` is set only after the real ActionGate
    APPROVE result and valid lease. Failure raises before any controller step.
    """
    if method not in {"B0", "Ours"}:
        raise ValueError("only B0/Ours recovery is supported")
    target = np.asarray(initial_public_eef_xyz, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("initial park target is not public XYZ")
    held = scene.public_grasp_latched()
    # No recovery command exists when an empty gripper is already at the
    # recorded public start pose. In particular, a denied first-round action
    # must not be converted into a synthetic move that can fail gate 2.
    if not held:
        current = np.asarray(scene.raw["robot0_eef_pos"], dtype=float)
        if current.shape != (3,) or not np.isfinite(current).all():
            raise ValueError("public recovery EEF pose is invalid")
        if float(np.linalg.norm(current - target)) <= .009:
            return RecoveryRecord("no_motion_already_at_public_start", 0,
                                  False, False, False), []
    gate = None
    lease = None
    if method == "Ours":
        if task_decision is None or reviewer is None or workspace is None or not round_id:
            raise ValueError("Ours recovery needs the real task decision/reviewer/workspace")
        current = workspace()  # current public RGB-D, never evaluator truth
        start = tuple(float(x) for x in current.eef_xyz)
        proposal = (PrimitiveProposal("hold", operated_entity="pan",
                                      changes_grasp_state=None)
                    if held else PrimitiveProposal(
                        "move_to", operated_entity=None,
                        motion_waypoints=(start, tuple(float(x) for x in target)),
                        gripper_radius_m=.035,
                        geometry_ambiguous=True,
                        parameters=(("recovery", "open_gripper_observation_park"),)))
        gate = ActionGate(
            round_id=f"{round_id}-recovery", task=task_decision,
            snapshot=workspace, reviewer=reviewer,
            finish=lambda _status, _message: None,
            timeout_s=180)
        decision = gate.check(proposal)
        if decision.result != "APPROVE" or not gate.is_current_lease(decision.lease):
            raise RuntimeError(f"S1 recovery gate 2 denied: {decision.reason}")
        lease = decision.lease
        adapter.set_commit_conditions(lease.commit_conditions)
    outlet = AdapterControlOutlet(adapter, action_gate_checked=gate is not None)
    try:
        if held:
            # Never force release or move toward the person's hands merely to
            # tidy the frame. One actual closed-gripper command establishes HOLD.
            action = np.zeros(7, dtype=np.float32)
            action[6] = 1.
            before = scene.sequence
            outlet.execute(action)
            if scene.sequence <= before or not scene.public_grasp_latched():
                raise RuntimeError("S1 held-pan safe HOLD was not physically maintained")
            record = RecoveryRecord("safe_hold", 1, True, gate is not None, True)
        else:
            record = park_to_public_start(scene, target, outlet)
    finally:
        if gate is not None:
            adapter.set_commit_conditions(None)
            gate.end_round()
    return record, [asdict(item) for item in gate.logs] if gate else []
