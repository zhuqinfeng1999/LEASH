"""Convert current public RGB-D frames to v4.1 gate evidence (no task truth)."""
from __future__ import annotations

from io import BytesIO
import time

import numpy as np
from PIL import Image

from research.v4.candidates.leash_commit_gate_20260924.commit_gate import (
    CameraFrame, PublicEvidence, PublicRobotState,
)
from research.v4.candidates.leash_public_capture_20260924.public_capture import (
    capture_public_packet,
)


def _png(rgb: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG")
    return output.getvalue()


def _depth_png(depth_m: np.ndarray) -> bytes:
    """Fixed 0–2 m color scale: green at 0 m, red at 2 m, magenta invalid."""
    depth = np.asarray(depth_m, dtype=float)
    valid = np.isfinite(depth) & (depth > 0)
    unit = np.clip(np.nan_to_num(depth, nan=0., posinf=2.) / 2., 0., 1.)
    colors = np.stack((255 * unit, 255 * (1. - unit),
                       np.full_like(unit, 90.)), axis=-1).astype(np.uint8)
    colors[~valid] = (255, 0, 255)
    return _png(colors)


def evidence_from_scene(scene, *, hold_started_ns: int | None = None,
                        hold_estimate: str = "unknown") -> PublicEvidence:
    """Read two fresh rendered cameras and public proprioception only.

    `hold_estimate` is a public sensor estimate; no simulator contact or object
    identity is inferred here.  Unknown forces the commit gate to fail closed.
    """
    start = time.monotonic_ns() if hold_started_ns is None else hold_started_ns
    packet = capture_public_packet(scene, hold_started_ns=start,
                                   max_capture_span_ns=2_000_000_000)
    frames = []
    for camera in (packet.agentview, packet.robot0_eye_in_hand):
        rgb = np.asarray(camera.rgb, dtype=np.uint8)
        depth = np.asarray(camera.depth_m, dtype=np.float32)
        if rgb.shape != (480, 640, 3) or depth.shape != (480, 640):
            raise ValueError("v4.2 public RGB-D must be actual 640x480")
        frames.append(CameraFrame(
            camera.name, _png(rgb), _depth_png(depth), 640, 480,
            {"K": np.asarray(camera.intrinsic_k).tolist(),
             "depth_color_scale_m": [0., 2.],
             "depth_colors": "green near; red far; magenta invalid"},
            {"camera_to_world": np.asarray(camera.extrinsic_cam2world).tolist()},
            camera.captured_ns,
        ))
    robot = PublicRobotState(
        gripper_closed=bool(np.asarray(packet.gripper_qpos_m).mean() < .005),
        hold_estimate=hold_estimate,
        observed_monotonic_ns=packet.observed_ns,
    )
    return PublicEvidence(*frames, robot, packet.observed_ns)
