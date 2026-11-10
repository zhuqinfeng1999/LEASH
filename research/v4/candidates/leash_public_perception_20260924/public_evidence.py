"""Conservative public-only perception building blocks for LEASH gate ③.

No function reads a MuJoCo scene, contact list, object transform, evaluator
state, spectator camera, or historical image in place of a missing current
image. `held` requires corroborated current two-view 3D object evidence and
independent visual co-motion; `empty` is deliberately not asserted by this
candidate. A separate certified gripper-gap visibility test is needed before
`empty` can safely bypass the commit reviewer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping

import numpy as np


CAMERAS = ("agentview", "robot0_eye_in_hand")


@dataclass(frozen=True)
class CameraObservation:
    name: str
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsic_k: np.ndarray
    extrinsic_cam2world: np.ndarray
    calibration_height: int
    calibration_width: int
    captured_ns: int


@dataclass(frozen=True)
class PublicPacket:
    agentview: CameraObservation
    robot0_eye_in_hand: CameraObservation
    eef_pos_m: np.ndarray
    eef_quat_xyzw: np.ndarray
    gripper_qpos_m: np.ndarray
    observed_ns: int


@dataclass(frozen=True)
class Validation:
    valid: bool
    reason: str


@dataclass(frozen=True)
class HeldEstimate:
    status: Literal["held", "empty", "unknown"]
    object_name: str | None
    reason: str
    current_center_world_m: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class TiltPrediction:
    current_upper_deg: float
    next_upper_deg: float


@dataclass(frozen=True)
class PublicDoorBox:
    """Heater door's conservatively padded world AABB from current RGB-D."""

    name: str | None
    lower_m: tuple[float, float, float]
    upper_m: tuple[float, float, float]
    source: Literal["public_rgbd"] = "public_rgbd"


@dataclass(frozen=True)
class DoorVerdict:
    status: Literal["clear", "possible_close", "unknown"]
    name: str | None
    reason: str


def validate_geometry(packet: PublicPacket) -> Validation:
    """Strict RGB-D/proprio/calibration shape check; no resizing or guessing."""
    if not isinstance(packet, PublicPacket):
        return Validation(False, "current public packet is missing")
    for expected, view in zip(CAMERAS, (packet.agentview, packet.robot0_eye_in_hand)):
        if not isinstance(view, CameraObservation) or view.name != expected:
            return Validation(False, f"{expected} camera is missing")
        rgb, depth = np.asarray(view.rgb), np.asarray(view.depth_m)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            return Validation(False, f"{expected} RGB is invalid")
        if depth.shape != rgb.shape[:2] or not np.issubdtype(depth.dtype, np.floating):
            return Validation(False, f"{expected} depth is not aligned to RGB")
        if not np.isfinite(depth).all() or not (depth > 0).all():
            return Validation(False, f"{expected} depth is missing or nonpositive")
        if (view.calibration_height, view.calibration_width) != rgb.shape[:2]:
            return Validation(False, f"{expected} calibration dimensions differ from RGB-D")
        k = np.asarray(view.intrinsic_k)
        e = np.asarray(view.extrinsic_cam2world)
        if k.shape != (3, 3) or e.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(e).all():
            return Validation(False, f"{expected} calibration is invalid")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(e[3], [0, 0, 0, 1], atol=1e-5):
            return Validation(False, f"{expected} calibration has invalid geometry")
        if not np.allclose(e[:3, :3].T @ e[:3, :3], np.eye(3), atol=1e-2):
            return Validation(False, f"{expected} extrinsic rotation is not rigid")
    if (np.asarray(packet.eef_pos_m).shape != (3,)
            or np.asarray(packet.eef_quat_xyzw).shape != (4,)
            or np.asarray(packet.gripper_qpos_m).shape != (2,)):
        return Validation(False, "public robot state has invalid shape")
    for x in (packet.eef_pos_m, packet.eef_quat_xyzw, packet.gripper_qpos_m):
        if not np.isfinite(x).all():
            return Validation(False, "public robot state is nonfinite")
    if not 0.99 <= np.linalg.norm(packet.eef_quat_xyzw) <= 1.01:
        return Validation(False, "public EEF quaternion is invalid")
    return Validation(True, "aligned public RGB-D and proprioception")


def validate_current(packet: PublicPacket, *, hold_started_ns: int,
                     max_camera_skew_ns: int = 100_000_000) -> Validation:
    spatial = validate_geometry(packet)
    if not spatial.valid:
        return spatial
    timestamps = (packet.agentview.captured_ns,
                  packet.robot0_eye_in_hand.captured_ns, packet.observed_ns)
    if any(ts < hold_started_ns for ts in timestamps):
        return Validation(False, "current evidence predates safety HOLD")
    if max(timestamps) - min(timestamps) > max_camera_skew_ns:
        return Validation(False, "public cameras/proprioception are not synchronized")
    return spatial


def _masked_center_world(view: CameraObservation, mask: np.ndarray,
                         *, min_pixels: int = 16) -> np.ndarray | None:
    mask = np.asarray(mask)
    if mask.shape != view.depth_m.shape or mask.dtype != np.bool_:
        return None
    rows, cols = np.nonzero(mask)
    if len(rows) < min_pixels:
        return None
    z = view.depth_m[rows, cols]
    k = view.intrinsic_k
    cam = np.stack(((cols - k[0, 2]) * z / k[0, 0],
                    (rows - k[1, 2]) * z / k[1, 1], z), axis=1)
    world = cam @ view.extrinsic_cam2world[:3, :3].T + view.extrinsic_cam2world[:3, 3]
    return np.median(world, axis=0)


def estimate_held_from_causal_visual_motion(
    *, current: PublicPacket, previous: PublicPacket | None,
    current_object_masks: Mapping[str, np.ndarray] | None,
    previous_object_masks: Mapping[str, np.ndarray] | None,
    public_object_name: str | None,
    hold_started_ns: int,
) -> HeldEstimate:
    """Assert only strong `held`; otherwise keep the safety-relevant unknown.

    Masks must come from a trusted *public-image* segmenter. A coincidental
    nearby object is insufficient: two cameras must agree in 3D, and the
    object must move rigidly with a lifted EEF over at least 5 cm. Previous
    frames are used only as co-motion evidence; current views are mandatory.
    """
    current_check = validate_current(current, hold_started_ns=hold_started_ns)
    if not current_check.valid:
        return HeldEstimate("unknown", None, current_check.reason)
    if previous is None or not validate_geometry(previous).valid or previous.observed_ns >= current.observed_ns:
        return HeldEstimate("unknown", None, "no valid prior public co-motion observation")
    if (not public_object_name or current_object_masks is None
            or previous_object_masks is None):
        return HeldEstimate("unknown", None, "public object segmentation is unavailable")
    centers: dict[str, list[np.ndarray]] = {"current": [], "previous": []}
    for label, packet, masks in (
        ("current", current, current_object_masks),
        ("previous", previous, previous_object_masks),
    ):
        for camera in CAMERAS:
            if camera not in masks:
                return HeldEstimate("unknown", None, f"{label} {camera} object mask missing")
            point = _masked_center_world(getattr(packet, camera), masks[camera])
            if point is None:
                return HeldEstimate("unknown", None, f"{label} {camera} object geometry ambiguous")
            centers[label].append(point)
        if np.linalg.norm(centers[label][0] - centers[label][1]) > 0.035:
            return HeldEstimate("unknown", None, f"{label} cameras disagree on object depth")
    now = np.mean(centers["current"], axis=0)
    before = np.mean(centers["previous"], axis=0)
    eef_delta = current.eef_pos_m - previous.eef_pos_m
    if np.linalg.norm(eef_delta) < 0.05 or eef_delta[2] < 0.025:
        return HeldEstimate("unknown", None, "no verified EEF lift and transport")
    if np.linalg.norm((now - before) - eef_delta) > 0.015:
        return HeldEstimate("unknown", None, "object did not co-move with the EEF")
    if np.linalg.norm(now - current.eef_pos_m) > 0.18:
        return HeldEstimate("unknown", None, "object is outside a conservative grasp envelope")
    opening = float(np.abs(current.gripper_qpos_m).sum())
    if not 0.004 <= opening <= 0.075:
        return HeldEstimate("unknown", None, "finger aperture does not corroborate holding")
    return HeldEstimate("held", public_object_name,
                        "two-view object geometry co-moves with lifted EEF",
                        tuple(float(v) for v in now))


def conservative_tilt_upper_bound(
    *, current_upper_deg: float | None, action_rot_xyz: np.ndarray,
    max_controller_rotation_deg_per_unit: float | None,
    model_error_deg: float = 2.0,
) -> TiltPrediction | None:
    """Return a certifiable bound, or None; never invent object orientation."""
    rot = np.asarray(action_rot_xyz, dtype=float)
    if (current_upper_deg is None or max_controller_rotation_deg_per_unit is None
            or rot.shape != (3,) or not np.isfinite(rot).all()
            or not math.isfinite(current_upper_deg)
            or not math.isfinite(max_controller_rotation_deg_per_unit)
            or not math.isfinite(model_error_deg)
            or min(current_upper_deg, max_controller_rotation_deg_per_unit, model_error_deg) < 0):
        return None
    next_upper = current_upper_deg + np.linalg.norm(rot) * max_controller_rotation_deg_per_unit + model_error_deg
    return TiltPrediction(float(current_upper_deg), float(next_upper))


def predict_heated_door_contact(
    *, packet: PublicPacket, action_xyz: np.ndarray,
    doors: tuple[PublicDoorBox, ...] | None,
    detector_coverage_certified: bool,
    max_axis_step_m: float | None,
    gripper_and_model_margin_m: float = 0.08,
) -> DoorVerdict:
    """Conservative one-step swept-AABB contact; contact may close the door.

    A `clear` result requires a complete public heater-door catalog and a
    calibrated upper bound on controller displacement. This does not infer
    closing direction; possible contact invokes gate ③'s reviewer. Unknown
    device identity must produce a BLOCK there.
    """
    spatial = validate_geometry(packet)
    if not spatial.valid:
        return DoorVerdict("unknown", None, spatial.reason)
    action = np.asarray(action_xyz, dtype=float)
    if (action.shape != (3,) or not np.isfinite(action).all()
            or max_axis_step_m is None or not math.isfinite(max_axis_step_m)
            or max_axis_step_m <= 0 or gripper_and_model_margin_m < 0):
        return DoorVerdict("unknown", None, "control sweep is not calibrated")
    if not detector_coverage_certified or doors is None:
        return DoorVerdict("unknown", None, "heated-door public detection coverage is not certified")
    start = np.asarray(packet.eef_pos_m, dtype=float)
    end = start + action * max_axis_step_m
    sweep_lo = np.minimum(start, end) - gripper_and_model_margin_m
    sweep_hi = np.maximum(start, end) + gripper_and_model_margin_m
    for door in doors:
        if door.source != "public_rgbd":
            return DoorVerdict("unknown", None, "door geometry is not public evidence")
        lo, hi = np.asarray(door.lower_m), np.asarray(door.upper_m)
        if (lo.shape != (3,) or hi.shape != (3,) or not np.isfinite(lo).all()
                or not np.isfinite(hi).all() or np.any(lo > hi)):
            return DoorVerdict("unknown", None, "door geometry is malformed")
        if np.all(sweep_hi >= lo) and np.all(sweep_lo <= hi):
            return DoorVerdict("possible_close" if door.name else "unknown",
                               door.name, "gripper sweep can contact a heated door")
    return DoorVerdict("clear", None, "certified public heater-door catalog is clear of sweep")
