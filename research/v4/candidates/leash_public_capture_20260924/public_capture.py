"""Current two-view RGB-D/proprioception capture for an isolated LEASH candidate.

Only the same native cameras that feed RPent are read. The simulator is used
solely to obtain those cameras' calibration, never object/contact/evaluator
state. Call on the scene's EGL owner thread after the gate-3 HOLD boundary.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from research.v4.candidates.leash_public_perception_20260924.public_evidence import (
    CAMERAS, CameraObservation, PublicPacket, validate_current,
)


Calibration = Callable[[object, str, int, int], tuple[np.ndarray, np.ndarray]]


def native_calibration(sim: object, name: str, height: int, width: int):
    """The online public camera model: K and OpenCV cam-to-world at exact RGB size."""
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix, get_camera_intrinsic_matrix,
    )
    return (np.asarray(get_camera_intrinsic_matrix(sim, name, height, width), dtype=float),
            np.asarray(get_camera_extrinsic_matrix(sim, name), dtype=float))


def capture_public_packet(
    scene: object, *, hold_started_ns: int,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    calibration: Calibration = native_calibration,
    max_capture_span_ns: int = 100_000_000,
) -> PublicPacket:
    """Capture fresh agent-facing evidence or fail closed with an exact reason.

    ``capture_started_ns`` is a conservative lower bound for every image. The
    scene is stationary between start/end because this function calls no
    controller step. A slow or stale capture is rejected, never relabeled as
    a newer image. The caller must keep motion blocked during this function.
    """
    if not isinstance(hold_started_ns, int) or hold_started_ns < 0:
        raise ValueError("HOLD boundary monotonic timestamp is missing")
    if not isinstance(max_capture_span_ns, int) or max_capture_span_ns <= 0:
        raise ValueError("capture span bound must be positive")
    started_ns = clock_ns()
    if started_ns < hold_started_ns:
        raise RuntimeError("public capture started before HOLD")
    packet = scene.capture_agent()
    completed_ns = clock_ns()
    if completed_ns < started_ns or completed_ns - started_ns > max_capture_span_ns:
        raise RuntimeError("public capture exceeded its freshness/skew bound")
    if not isinstance(packet, dict) or not isinstance(packet.get("cameras"), dict):
        raise RuntimeError("current public camera packet is malformed")
    cameras = packet["cameras"]
    if set(cameras) != set(CAMERAS):
        raise RuntimeError("public packet must contain exactly agentview and robot0_eye_in_hand")
    sim = scene.env.sim  # Public camera calibration only; no truth/contacts.
    built = {}
    for name in CAMERAS:
        entry = cameras[name]
        if not isinstance(entry, dict) or set(entry) != {"rgb", "depth_m"}:
            raise RuntimeError(f"{name} RGB-D payload is malformed")
        rgb = np.asarray(entry["rgb"])
        depth = np.asarray(entry["depth_m"])
        if rgb.ndim != 3 or depth.shape != rgb.shape[:2]:
            raise RuntimeError(f"{name} RGB-D dimensions disagree")
        height, width = rgb.shape[:2]
        intrinsic, extrinsic = calibration(sim, name, height, width)
        built[name] = CameraObservation(
            name=name, rgb=rgb.copy(), depth_m=depth.copy(),
            intrinsic_k=np.asarray(intrinsic, dtype=float).copy(),
            extrinsic_cam2world=np.asarray(extrinsic, dtype=float).copy(),
            calibration_height=height, calibration_width=width,
            captured_ns=started_ns,
        )
    result = PublicPacket(
        agentview=built["agentview"],
        robot0_eye_in_hand=built["robot0_eye_in_hand"],
        eef_pos_m=np.asarray(packet["eef_pos"], dtype=float).copy(),
        eef_quat_xyzw=np.asarray(packet["eef_quat"], dtype=float).copy(),
        gripper_qpos_m=np.asarray(packet["gripper_qpos"], dtype=float).copy(),
        observed_ns=completed_ns,
    )
    checked = validate_current(result, hold_started_ns=hold_started_ns,
                               max_camera_skew_ns=max_capture_span_ns)
    if not checked.valid:
        raise RuntimeError(f"invalid fresh public evidence: {checked.reason}")
    return result
