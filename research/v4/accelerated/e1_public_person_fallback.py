"""Current RGB-D support check for a previously seen stationary person box.

E1 moves a hand while the seated torso stays put. If text segmentation loses
the person after the arm enters the plate, this check can retain the last
publicly observed torso envelope only while fresh agentview depth still fills
that same 3-D volume. The moving hand is always localized separately.
"""
from __future__ import annotations

import numpy as np


def current_box_depth_support(camera: dict, box, *, stride: int = 4) -> int:
    depth = np.asarray(camera["depth_m"], dtype=float)
    k = np.asarray(camera["intrinsics"], dtype=float)
    e = np.asarray(camera["extrinsics"], dtype=float)
    if (depth.shape != (480, 640) or k.shape != (3, 3) or e.shape != (4, 4) or
            not np.isfinite(depth).all() or not (depth > 0).all() or
            not np.isfinite(k).all() or not np.isfinite(e).all() or
            abs(np.linalg.det(k)) < 1e-9 or stride < 1):
        raise ValueError("fresh public RGB-D calibration is invalid")
    ys, xs = np.mgrid[0:480:stride, 0:640:stride]
    z = depth[::stride, ::stride]
    cam = np.stack(((xs-k[0, 2])*z/k[0, 0],
                    (ys-k[1, 2])*z/k[1, 1], z), axis=-1).reshape(-1, 3)
    world = cam @ e[:3, :3].T + e[:3, 3]
    lower, upper = np.asarray(box.lower), np.asarray(box.upper)
    if (lower.shape != (3,) or upper.shape != (3,) or
            not np.isfinite(lower).all() or not np.isfinite(upper).all() or
            np.any(lower >= upper)):
        raise ValueError("cached public person box is invalid")
    return int(np.count_nonzero(np.all((world >= lower) & (world <= upper), axis=1)))


def person_box_supported_now(camera: dict, box, baseline_support: int) -> tuple[bool, int]:
    if not isinstance(baseline_support, int) or baseline_support < 400:
        raise ValueError("person baseline has insufficient public depth support")
    current = current_box_depth_support(camera, box)
    return current >= max(400, int(.6 * baseline_support)), current
