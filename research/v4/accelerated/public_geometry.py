"""Derive conservative world boxes from public RGB-D and frozen SAM3 masks.

This module never reads simulator object poses, evaluator markers, grasp zones,
or spectator images. Missing/ambiguous segmentation returns no box and the
existing action gate fails closed on a named forbidden zone.
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import PublicBox


class PublicGeometryInvalid(ValueError):
    """Aligned perception output fails geometry quality; not absent geometry."""
    def __init__(self, lower, upper, nonfinite, *, mask=None,depth=None,k=None,t=None,
                 valid_pixels=None):
        self.lower = None if lower is None else np.asarray(lower)
        self.upper = None if upper is None else np.asarray(upper)
        self.nonfinite = nonfinite
        self.public_inputs = (mask,depth,k,t)
        self.valid_pixels = valid_pixels
        super().__init__('segmented entity geometry is implausible' if valid_pixels is None
                         else 'too few valid public mask/depth pixels')


def world_box_from_mask(*, name: str, mask: np.ndarray, depth_m: np.ndarray,
                        intrinsic_k: np.ndarray, extrinsic_cam2world: np.ndarray,
                        margin_m: float = .03) -> PublicBox:
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=float)
    k = np.asarray(intrinsic_k, dtype=float)
    e = np.asarray(extrinsic_cam2world, dtype=float)
    if mask.shape != depth.shape or mask.ndim != 2 or k.shape != (3, 3) or e.shape != (4, 4):
        raise ValueError("public mask, aligned depth or calibration is malformed")
    if not name or not np.isfinite(k).all() or not np.isfinite(e).all() or abs(np.linalg.det(k)) < 1e-9:
        raise ValueError("public box name/calibration invalid")
    ys, xs = np.nonzero(mask & np.isfinite(depth) & (depth > 0) & (depth < 3))
    if len(xs) < 80:
        # Aligned, well-formed perception with insufficient depth is unresolved
        # geometry, not a malformed request or a service failure. No box exists.
        raise PublicGeometryInvalid(None,None,int((mask & ~np.isfinite(depth)).sum()),
            mask=mask,depth=depth,k=k,t=e,valid_pixels=len(xs))
    # Deterministic sampling only bounds compute; no simulator state enters.
    if len(xs) > 12000:
        select = np.linspace(0, len(xs)-1, 12000, dtype=int)
        ys, xs = ys[select], xs[select]
    z = depth[ys, xs]
    # A few SAM edge pixels can inherit the far background depth. Keep the
    # observed foreground cluster when the far tail is both sparse and clearly
    # separated; never trim a substantial second depth surface. This uses only
    # the aligned public depth image and remains source-versioned for new runs.
    far_bleed = z > float(np.median(z)) + .35
    if 0 < int(far_bleed.sum()) <= max(1, int(.01 * len(z))):
        keep = ~far_bleed
        ys, xs, z = ys[keep], xs[keep], z[keep]
    camera = np.linalg.inv(k) @ np.vstack((xs*z, ys*z, z))
    world = (e @ np.vstack((camera, np.ones_like(z))))[:3].T
    lower = np.quantile(world, .005, axis=0) - margin_m
    upper = np.quantile(world, .995, axis=0) + margin_m
    if not np.isfinite(world).all() or np.any(upper-lower > 1.2):
        raise PublicGeometryInvalid(lower,upper,int((~np.isfinite(world)).sum()),
            mask=mask,depth=depth,k=k,t=e)
    return PublicBox(name, tuple(map(float, lower)), tuple(map(float, upper)))


def sam_public_box(*, name: str, prompt: str, camera: dict, sam3_client,
                   min_score: float = .2) -> tuple[PublicBox, dict]:
    """One public RGB-D SAM3 query; return box and auditable mask statistics."""
    rgb = np.asarray(camera["rgb"], dtype=np.uint8)
    depth = np.asarray(camera["depth_m"], dtype=float)
    if rgb.shape != (*depth.shape, 3):
        raise ValueError("public RGB-D misaligned")
    buffer = BytesIO(); Image.fromarray(rgb).save(buffer, format="PNG")
    result = sam3_client.segment(buffer.getvalue(), text_prompt=prompt,
                                 min_score=min_score)
    if not result.found or result.mask is None:
        raise ValueError("SAM3 did not locate the named public entity")
    box = world_box_from_mask(name=name, mask=result.mask, depth_m=depth,
                              intrinsic_k=np.asarray(camera["intrinsics"]),
                              extrinsic_cam2world=np.asarray(camera["extrinsics"]))
    return box, {"name": name, "prompt": prompt, "score": result.score,
                 "mask_pixels": int(np.asarray(result.mask).sum()),
                 "box_lower": box.lower, "box_upper": box.upper,
                 "source": "public_rgbd_perception"}
