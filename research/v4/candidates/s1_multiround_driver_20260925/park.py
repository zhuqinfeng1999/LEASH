"""Physical, open-gripper S1 inter-round observation park.

The caller injects the method's already-reviewed low-level control outlet.
No simulator reset, pose edit, release, or grasp-truth access is performed.
"""
from __future__ import annotations

from typing import Protocol
import numpy as np

from .bridge import RecoveryRecord


class ParkController(Protocol):
    is_common_outlet: bool
    action_gate_checked: bool

    def execute(self, action: np.ndarray) -> None: ...


def park_to_public_start(scene, target_public_eef_xyz, controller: ParkController,
                         *, max_steps: int = 260) -> RecoveryRecord:
    """Return the empty, open gripper to a saved public EEF pose physically."""
    target = np.asarray(target_public_eef_xyz, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Park target must be a finite public EEF XYZ")
    if not controller.is_common_outlet:
        raise ValueError("Recovery cannot bypass the common controller outlet")
    if scene.public_grasp_latched():
        raise RuntimeError("Cannot park while holding; task recovery must resolve hold safely")
    steps = 0
    for _ in range(max_steps):
        current = np.asarray(scene.raw["robot0_eef_pos"], dtype=float)
        delta = target - current
        if float(np.linalg.norm(delta)) <= .009:
            break
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(delta, -.010, .010) / .05
        action[6] = -1.0  # already empty; keep gripper open
        previous_sequence = scene.sequence
        controller.execute(action)
        if scene.sequence <= previous_sequence:
            raise RuntimeError("Recovery command did not advance real physics")
        if scene.public_grasp_latched():
            raise RuntimeError("Recovery unexpectedly attached an object")
        steps += 1
    miss = float(np.linalg.norm(target - np.asarray(scene.raw["robot0_eef_pos"], dtype=float)))
    if miss > .022:
        raise RuntimeError(f"Physical observation park missed target by {miss:.3f} m")
    return RecoveryRecord(
        kind="physical_open_gripper_observation_park",
        actual_control_steps=steps,
        passed_common_outlet=True,
        action_gate_checked=bool(controller.action_gate_checked),
        final_public_grasp_latched=scene.public_grasp_latched(),
    )
