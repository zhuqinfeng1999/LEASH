"""Recorded-image SAM3; missing metric calibration does not disable RGB masks."""
import numpy as np

from robots.libero.tools import (
    _make_segment_overlay, _mask_to_world, _next_segment_index,
)


def _not_found_recovery(state, image_name, camera, step):
    """Describe existing readers without guessing a pixel or route clearance."""
    size = None
    try:
        image = np.asarray(state.load(image_name, step=step))
        if image.ndim == 3 and image.shape[2] == 3:
            size = list(image.shape[:2])
    except (OSError, ValueError, KeyError):
        pass  # A diagnostic must not replace the original not-found result.
    return dict(kind="perception_not_found", native_image_size_hw=size,
        point_input_convention="[row, col] in this recorded camera's native image",
        point_segmentation_call=dict(name="segment", camera=camera, step=step,
            point="Select [row, col] on a visible part of the intended object; omit prompt."),
        alternative="back_project can query selected visible pixels in this same recorded view.",
        scope="No mask was found by this request. This does not assess whether a transport route exists or certify its clearance.",
        readonly=True, physical_commands=0)


def segment_recorded(client, state, *, prompt="", camera="agentview", step=-1,
                     point=None, min_score=0.2):
    """Preserve the reference SAM3 selection and XYZ aggregation conventions."""
    if camera not in ("agentview", "wrist"):
        return {"error": f"unknown segment camera: {camera}"}
    prompt = prompt.strip()
    if bool(prompt) == (point is not None):
        return {"error": "segment needs exactly one of prompt or point"}
    try:
        record = state.get(step)
        nn = record.step_idx
        image_name = next((name for name in (camera + "_high.png", camera + ".png")
                           if state.exists(name, step=nn)), None)
        if image_name is None:
            return {"error": "Recorded RGB image unavailable", "camera": camera}
        world_name = camera + ("_world_high.npz" if "_high" in image_name else "_world.npz")
        data = client.segment(state.load_bytes(image_name, step=nn),
                              text_prompt=prompt or None, point=point, min_score=min_score)
    except Exception as exc:
        return {"error": f"segmentation service call failed: {exc}", "camera": camera,
                "fallback": "Use manual visual localization and back_project."}
    index = _next_segment_index(record)
    blob = dict(found=data.found, mode="text" if prompt else "point", camera=camera,
                source_step=nn, segment_index=index, image_artifact=image_name,
                min_score=min_score, score=round(float(data.score), 3) if data.score is not None else None,
                box=data.box, mask_shape=list(data.mask_shape) if data.mask_shape else None,
                world_xyz=None)
    blob["prompt" if prompt else "point"] = prompt or point
    overlay_path = None
    if data.found and isinstance(data.mask, np.ndarray):
        mask = data.mask.astype(bool)
        rgb = state.load(image_name, step=nn)
        if mask.shape != rgb.shape[:2]:
            return {"error": "SAM3 mask/image shape mismatch; no resizing performed",
                    "found": False, "world_xyz": None}
        rows, cols = np.nonzero(mask)
        blob.update(n_pixels=int(mask.sum()),
                    centroid_pixel=[int(round(float(np.median(cols)))),
                                    int(round(float(np.median(rows))))] if rows.size else None,
                    centroid_pixel_convention="[col, row]")
        try:
            if not state.exists(world_name, step=nn):
                raise ValueError("Metric world map unavailable for this recorded camera")
            blob.update(_mask_to_world(mask, state.load(world_name, step=nn)))
            blob["world_artifact"] = world_name
        except Exception as exc:
            blob.update(world_xyz=None, world_error=str(exc), n_valid=0)
        # Keep the mask itself for deterministic audit, not just the overlay.
        blob["mask_artifact"] = state.save(f"segment_mask_{index:02d}.png",
                                           mask.astype(np.uint8) * 255, step=nn)
        overlay = _make_segment_overlay(rgb, mask)
        if overlay is not None:
            overlay_path = state.save(f"segment_overlay_{index:02d}.png", overlay, step=nn)
    else:
        blob.update(error=data.reason or "SAM3 found no mask",
                    world_error=data.reason or "SAM3 found no mask")
    saved = state.save(f"segment_{index:02d}.json", blob, step=nn)
    result = dict(blob, step=nn, segment_artifact=saved)
    if "error" in blob:
        result["fallback"] = "Use manual visual localization and back_project."
        result["localization_recovery"] = _not_found_recovery(state, image_name, camera, nn)
    if overlay_path:
        result["overlay_artifact"] = overlay_path
        result["_image_bytes"] = state.load_bytes(overlay_path, step=nn)
    return result


def configured_client(config):
    """Borrow the explicitly configured SAM3 RPC service; never start a device."""
    endpoint = (config or {}).get("sam3_endpoint")
    if not endpoint:
        return None
    from urllib.parse import urlsplit
    from rpent.utils.http_rpc import HttpRpcClient
    from rpent.utils.sam3_client import Sam3Client
    parsed = urlsplit(endpoint)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("sam3_endpoint must be an http RPC URL without credentials")
    return Sam3Client(HttpRpcClient(endpoint), timeout_s=120)
