"""Public, rigid-grasp P3 cup envelope for gate-2 path checks.

This is an isolated candidate.  It deliberately does not read MuJoCo object
poses, grasp regions, evaluator state, or a spectator camera.  The shared
public latch means that the same weld remains active for either method.  A
rigid body's points cannot move relative to that weld until it is released.
The envelope is a sphere around the gripper, so every possible wrist rotation
is covered; nevertheless explicit rotation primitives invalidate this v1
calibration and require a fresh public visual qualification.

The source and first-held boxes must be separately segmented from 640x480
public RGB-D.  They already include the known 3 cm box-construction margin.
Only the known margin is removed before computing the cuboid corner bound.
Subsequent SAM boxes are *not* used to shrink or inflate this bound: #193's
partially occluded cup mask included unrelated depth and expanded 7 to 18 cm.
Two independently grounded, agreeing public-view boxes that contradict the
rigid model invalidate it rather than silently override it.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np

from research.v4.accelerated.p3_review_names import (
    PublicSweptRadius, UnresolvedPublicHazard,
)
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import PublicBox


KNOWN_BOX_MARGIN_M = .03
CONTROL_ALLOWANCE_M = .008
GRIPPER_RADIUS_M = .035
ROTATION_PRIMITIVES = frozenset({
    "rotate_wrist", "rotate_pitch", "move_pose", "pi0_pick", "pi0_doubled",
})


def _vec(value, n: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise UnresolvedPublicHazard(f"invalid public {name}")
    return result


def _public_cup_box(box: PublicBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if box.name != "cup" or box.source != "public_rgbd_perception":
        raise UnresolvedPublicHazard("cup box is not public RGB-D perception")
    lo, hi = _vec(box.lower, 3, "cup lower"), _vec(box.upper, 3, "cup upper")
    if not np.all(hi > lo):
        raise UnresolvedPublicHazard("cup box is empty")
    half = (hi - lo) / 2 - KNOWN_BOX_MARGIN_M
    if (np.any(half <= 0) or np.any(half > .12) or
            not (.02 <= 2*half[0] <= .20) or
            not (.02 <= 2*half[1] <= .20) or
            not (.04 <= 2*half[2] <= .24)):
        raise UnresolvedPublicHazard("public cup footprint is not qualified")
    return lo, hi, half


@dataclass
class HeldCupSweep:
    """One grasp's immutable envelope, invalidated on release or rotation.

    The caller supplies the current public latch at *every* primitive start.
    `grasp_epoch` is a local counter that increments at every public false->true
    latch transition; no simulator object identity is read.  A new grasp must
    arm a new instance.  The caller must call `before_primitive` for every
    robot primitive, even those that gate 2 later denies.
    """

    radius: PublicSweptRadius
    grasp_epoch: int
    first_held_frame_id: int
    first_eef_xyz: tuple[float, float, float]
    first_cup_center_xyz: tuple[float, float, float]
    last_frame_id: int
    last_eef_xyz: tuple[float, float, float]
    valid: bool = True
    invalid_reason: str | None = None

    @classmethod
    def arm(cls, *, source_box: PublicBox, first_held_box: PublicBox,
            first_eef_xyz, first_eef_quat, source_frame_id: int,
            first_held_frame_id: int, grasp_epoch: int,
            public_latch: bool, gripper_closed: bool,
            source_score: float, first_held_score: float,
            source_mask_pixels: int, first_held_mask_pixels: int
            ) -> "HeldCupSweep":
        if not public_latch or not gripper_closed:
            raise UnresolvedPublicHazard("rigid held-cup latch not confirmed")
        if (not isinstance(source_frame_id, int) or
                not isinstance(first_held_frame_id, int) or
                source_frame_id >= first_held_frame_id or
                not isinstance(grasp_epoch, int) or grasp_epoch < 1):
            raise UnresolvedPublicHazard("public frame or latch epoch invalid")
        if (min(source_score, first_held_score) < .2 or
                min(source_mask_pixels, first_held_mask_pixels) < 80):
            raise UnresolvedPublicHazard("public cup segmentation too weak")
        source_lo, source_hi, source_half = _public_cup_box(source_box)
        held_lo, held_hi, held_half = _public_cup_box(first_held_box)
        source_center = (source_lo + source_hi) / 2
        held_center = (held_lo + held_hi) / 2
        # Arming occurs immediately after closure, before a transport.  A
        # distant first-held detection may be a different object or stale mask.
        if np.linalg.norm(source_center - held_center) > .06:
            raise UnresolvedPublicHazard("first-held cup is not source cup")
        eef = _vec(first_eef_xyz, 3, "end-effector position")
        quat = _vec(first_eef_quat, 4, "end-effector quaternion")
        if abs(np.linalg.norm(quat) - 1.) > .05:
            raise UnresolvedPublicHazard("end-effector quaternion not normalized")
        center_offset = held_center - eef
        if np.linalg.norm(center_offset) > .08:
            raise UnresolvedPublicHazard("cup is too far from public gripper")
        half = np.maximum(source_half, held_half)
        # All eight cuboid corners are included.  A sphere over these corners
        # stays valid under arbitrary rigid wrist rotation without needing the
        # hidden object transform or trusting later occluded SAM masks.
        radius = max(float(np.linalg.norm(center_offset + half*np.array(sign)))
                     for sign in product((-1., 1.), repeat=3))
        radius += CONTROL_ALLOWANCE_M
        total = max(GRIPPER_RADIUS_M, radius)
        if total > .16:
            raise UnresolvedPublicHazard("held cup envelope too large to certify")
        xy_half = tuple(map(float, half[:2]))
        sweep = PublicSweptRadius(
            GRIPPER_RADIUS_M, max(0., total-GRIPPER_RADIUS_M), total,
            xy_half, float(np.linalg.norm(half[:2])),
            float(np.linalg.norm(center_offset[:2])), CONTROL_ALLOWANCE_M,
            KNOWN_BOX_MARGIN_M)
        return cls(sweep, grasp_epoch, first_held_frame_id,
                   tuple(map(float, eef)), tuple(map(float, held_center)),
                   first_held_frame_id, tuple(map(float, eef)))

    def before_primitive(self, name: str, *, arguments: dict | None = None,
                         public_latch: bool,
                         gripper_closed: bool, grasp_epoch: int,
                         current_frame_id: int, current_eef_xyz,
                         current_eef_quat,
                         corroborating_cup_boxes: Iterable[tuple[str, PublicBox]] = ()
                         ) -> PublicSweptRadius:
        """Return the frozen conservative radius or fail closed.

        A single later SAM box is allowed to be noisy/occluded.  Two public
        camera boxes must agree on a contradictory center before declaring
        the rigid sensor inconsistent.  Absence of visible cup is tolerated
        only while the common weld latch remains active.
        """
        if not self.valid:
            raise UnresolvedPublicHazard(self.invalid_reason or "held sweep invalid")
        if (name in ROTATION_PRIMITIVES or
                name == "move_to" and isinstance(arguments, dict)
                and arguments.get("target_yaw") is not None):
            # This is a proposed action, not an executed state transition.
            # Deny it without poisoning the already qualified held model; a
            # later straight-line or HOLD proposal may still be safe.
            raise UnresolvedPublicHazard("rotation primitive requires new qualification")
        if not public_latch or not gripper_closed or grasp_epoch != self.grasp_epoch:
            return self._invalidate("public grasp latch lost or new grasp began")
        if (not isinstance(current_frame_id, int) or
                current_frame_id < self.first_held_frame_id):
            return self._invalidate("public frame older than grasp")
        current = _vec(current_eef_xyz, 3, "end-effector position")
        q = _vec(current_eef_quat, 4, "end-effector quaternion")
        if abs(np.linalg.norm(q) - 1.) > .05:
            return self._invalidate("end-effector quaternion invalid")
        if (np.linalg.norm(current-np.asarray(self.last_eef_xyz)) > .002
                and current_frame_id <= self.last_frame_id):
            # Missing freshness is recoverable by a new public RGB-D render.
            # No controller instruction has been approved at this boundary.
            raise UnresolvedPublicHazard("public RGB-D frame stale after robot motion")
        self.last_frame_id = current_frame_id
        self.last_eef_xyz = tuple(map(float, current))
        # The conservative sphere is invariant to rotation, but a public
        # visual contradiction from BOTH views can reveal latch/sensor error.
        contradictory: dict[str, np.ndarray] = {}
        seen: set[str] = set()
        for camera, box in corroborating_cup_boxes:
            if camera not in {"agentview", "robot0_eye_in_hand"} or camera in seen:
                return self._invalidate("public cup evidence has duplicate/unknown camera")
            seen.add(camera)
            if box.name != "cup" or box.source != "public_rgbd_perception":
                return self._invalidate("corroborating cup box is not public RGB-D")
            try:
                lo, hi, _ = _public_cup_box(box)
            except UnresolvedPublicHazard:
                # A *later* single-view SAM mask can include occluding robot
                # pixels. It cannot enlarge the frozen grasp-time envelope or
                # prove a displaced cup. Only two valid agreeing views can
                # contradict the rigid public latch.
                continue
            center = (lo + hi) / 2
            if np.linalg.norm(center-current) > self.radius.total_radius_m + .03:
                contradictory[camera] = center
        if (len(contradictory) == 2 and np.linalg.norm(
                contradictory["agentview"] -
                contradictory["robot0_eye_in_hand"]) < .03):
            return self._invalidate("two public views contradict rigid cup latch")
        return self.radius

    def _invalidate(self, reason: str):
        self.valid, self.invalid_reason = False, reason
        raise UnresolvedPublicHazard(reason)
