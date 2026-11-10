"""v4.2 public observation contract, independent of a particular simulator.

Call ``public_observation`` once per simulator frame, with RGB and metric depth
from that same render. Spectator pixels and evaluator state are never returned.
The optional pi0.5 view is resampled from this packet, never rendered anew.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


PUBLIC_CAMERAS = ("agentview", "robot0_eye_in_hand")
PUBLIC_HEIGHT = 480
PUBLIC_WIDTH = 640
ROBOT_STATE_KEYS = frozenset({"eef_pos", "eef_quat", "gripper_qpos"})


def _camera(name: str, frame: Mapping, frame_id: int) -> dict:
    required = {"rgb", "depth_m", "intrinsics", "extrinsics", "frame_id"}
    if not isinstance(frame, Mapping) or set(frame) != required:
        raise ValueError(f"{name}: camera fields must be exactly {sorted(required)}")
    if frame["frame_id"] != frame_id:
        raise ValueError(f"{name}: stale or unmatched render frame")
    rgb = np.asarray(frame["rgb"])
    depth = np.asarray(frame["depth_m"])
    k = np.asarray(frame["intrinsics"], dtype=float)
    pose = np.asarray(frame["extrinsics"], dtype=float)
    if rgb.shape != (PUBLIC_HEIGHT, PUBLIC_WIDTH, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"{name}: RGB must be uint8 480x640x3")
    if depth.shape != (PUBLIC_HEIGHT, PUBLIC_WIDTH) or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError(f"{name}: metric depth must be floating 480x640")
    if not np.isfinite(depth).all() or not (depth > 0).all():
        raise ValueError(f"{name}: invalid metric depth")
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError(f"{name}: invalid camera intrinsics")
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError(f"{name}: invalid camera extrinsics")
    # Copy, so a renderer's reused buffer cannot mutate the archived evidence.
    return {"rgb": rgb.copy(), "depth_m": depth.copy(), "intrinsics": k.copy(),
            "extrinsics": pose.copy(), "frame_id": frame_id}


def public_observation(rendered: Mapping, robot_state: Mapping, *, frame_id: int) -> dict:
    """Accept exactly two fresh agent frames; ignore a separately stored spectator.

    ``rendered`` must contain only the two public cameras. A caller that also
    renders a spectator must strip it before this boundary; accidental leakage
    therefore fails rather than being silently passed onward.
    """
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError("frame_id must be a nonnegative integer")
    if not isinstance(rendered, Mapping) or set(rendered) != set(PUBLIC_CAMERAS):
        raise ValueError("online packet requires exactly agentview and robot0_eye_in_hand")
    if not isinstance(robot_state, Mapping) or set(robot_state) != ROBOT_STATE_KEYS:
        raise ValueError("online robot state contains private or missing fields")
    robot = {name: np.asarray(robot_state[name], dtype=float).copy() for name in ROBOT_STATE_KEYS}
    if (robot["eef_pos"].shape != (3,) or robot["eef_quat"].shape != (4,)
            or robot["gripper_qpos"].ndim != 1 or not robot["gripper_qpos"].size
            or any(not np.isfinite(value).all() for value in robot.values())):
        raise ValueError("invalid public proprioception")
    return {"frame_id": frame_id,
            "cameras": {name: _camera(name, rendered[name], frame_id) for name in PUBLIC_CAMERAS},
            "robot_state": robot}


def downsample_for_pi05(public_packet: Mapping, *, height: int, width: int) -> dict:
    """Derive a π0.5 RGB-D packet from the *same* public render and calibration.

    Nearest source pixels keep depth metric and avoid interpolation through
    foreground edges. This helper does not select π0.5's model input size.
    """
    if not (isinstance(height, int) and isinstance(width, int)
            and 0 < height <= PUBLIC_HEIGHT and 0 < width <= PUBLIC_WIDTH):
        raise ValueError("π0.5 resolution must be positive and no larger than public RGB-D")
    if not isinstance(public_packet, Mapping) or set(public_packet) != {"frame_id", "cameras", "robot_state"}:
        raise ValueError("invalid public packet")
    frame_id = public_packet["frame_id"]
    cameras = public_packet["cameras"]
    if set(cameras) != set(PUBLIC_CAMERAS):
        raise ValueError("π0.5 packet must derive from both public cameras")
    rows = np.minimum(((np.arange(height) + .5) * PUBLIC_HEIGHT / height).astype(int), PUBLIC_HEIGHT - 1)
    cols = np.minimum(((np.arange(width) + .5) * PUBLIC_WIDTH / width).astype(int), PUBLIC_WIDTH - 1)
    output = {}
    for name in PUBLIC_CAMERAS:
        source = cameras[name]
        if source.get("frame_id") != frame_id or np.asarray(source["rgb"]).shape != (PUBLIC_HEIGHT, PUBLIC_WIDTH, 3) or np.asarray(source["depth_m"]).shape != (PUBLIC_HEIGHT, PUBLIC_WIDTH):
            raise ValueError(f"{name}: not a validated full-resolution source frame")
        k = np.asarray(source["intrinsics"], dtype=float).copy()
        sx, sy = width / PUBLIC_WIDTH, height / PUBLIC_HEIGHT
        k[0, 0] *= sx
        k[1, 1] *= sy
        k[0, 2] = sx * (k[0, 2] + .5) - .5
        k[1, 2] = sy * (k[1, 2] + .5) - .5
        output[name] = {"rgb": np.asarray(source["rgb"])[np.ix_(rows, cols)].copy(),
                        "depth_m": np.asarray(source["depth_m"])[np.ix_(rows, cols)].copy(),
                        "intrinsics": k, "extrinsics": np.asarray(source["extrinsics"]).copy(),
                        "frame_id": frame_id}
    return {"frame_id": frame_id, "cameras": output,
            "robot_state": {key: value.copy() for key, value in public_packet["robot_state"].items()}}


@dataclass(frozen=True)
class VisibilityQA:
    name: str
    camera: str
    pixel_width: int
    pixel_height: int
    minimum_px: int
    size_pass: bool
    reason: str


def audit_segmentation(name: str, camera: str, mask: np.ndarray | None,
                       *, minimum_width_px: int = 15) -> VisibilityQA:
    """Check a key object's rendered width; missing masks fail honestly."""
    if camera not in PUBLIC_CAMERAS or minimum_width_px <= 0:
        raise ValueError("invalid camera or width threshold")
    if mask is None:
        return VisibilityQA(name, camera, 0, 0, minimum_width_px, False, "mask_missing")
    mask = np.asarray(mask)
    if mask.shape != (PUBLIC_HEIGHT, PUBLIC_WIDTH) or mask.dtype != np.bool_:
        return VisibilityQA(name, camera, 0, 0, minimum_width_px, False, "invalid_mask")
    ys, xs = np.where(mask)
    if not len(xs):
        return VisibilityQA(name, camera, 0, 0, minimum_width_px, False, "not_visible")
    w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    return VisibilityQA(name, camera, w, h, minimum_width_px,
                        w >= minimum_width_px,
                        "size_pass" if w >= minimum_width_px else "too_small")


def audit_text_glyphs(name: str, glyph_masks: list[np.ndarray] | None,
                      *, minimum_height_px: int = 12) -> dict:
    """Pixel-height check only. Readability still requires OCR or human QA."""
    if minimum_height_px <= 0:
        raise ValueError("invalid glyph threshold")
    if not glyph_masks:
        return {"name": name, "minimum_height_px": minimum_height_px,
                "min_observed_height_px": 0, "size_pass": False,
                "readability_verified": False, "reason": "glyph_masks_missing"}
    heights = []
    for mask in glyph_masks:
        mask = np.asarray(mask)
        if mask.shape != (PUBLIC_HEIGHT, PUBLIC_WIDTH) or mask.dtype != np.bool_:
            return {"name": name, "minimum_height_px": minimum_height_px,
                    "min_observed_height_px": 0, "size_pass": False,
                    "readability_verified": False, "reason": "invalid_glyph_mask"}
        ys = np.where(mask)[0]
        if not len(ys):
            return {"name": name, "minimum_height_px": minimum_height_px,
                    "min_observed_height_px": 0, "size_pass": False,
                    "readability_verified": False, "reason": "glyph_not_visible"}
        heights.append(int(ys.max() - ys.min() + 1))
    smallest = min(heights)
    return {"name": name, "minimum_height_px": minimum_height_px,
            "min_observed_height_px": smallest, "size_pass": smallest >= minimum_height_px,
            "readability_verified": False,
            "reason": "size_pass_readability_unverified" if smallest >= minimum_height_px else "glyph_too_small"}
