"""Candidate E1 knife pick from the two declared public RGB-D cameras.

Only the visible red handle and public gripper latch guide the local pickup.
Every child primitive goes through the supplied B0/Ours toolkit boundary. This
does not move to the plate, release an object, or use evaluator / grasp-site
coordinates. The red-handle appearance and contact offset are declared tool
priors shared by both methods, not instance-specific hidden state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import nullcontext
from typing import Any, Callable

import numpy as np
from scipy.ndimage import label

from research.v3.bridge.b0 import _tool_result
from research.v4.accelerated.public_geometry import world_box_from_mask


PICK_SPEC = {
    "name": "pick",
    "description": (
        "Locate the visible red knife handle in current public RGB-D and "
        "perform a bounded local pickup. Six normal RPent child primitives "
        "consume the shared action budget and pass through normal safety gates. "
        "The tool stops on denial or an empty grasp; it does not transport, "
        "release, or place the knife."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"], "additionalProperties": False,
    },
}


class KnifePerceptionError(ValueError):
    pass


@dataclass(frozen=True)
class KnifePickEstimate:
    xyz_m: tuple[float, float, float]
    frame_id: int
    mask_pixels: int
    visible_span_m: tuple[float, float, float]
    method: str = "public_red_handle_depth_quantiles_v1"


def _public_handle_mask(rgb: np.ndarray) -> np.ndarray:
    """Find a unique contiguous dark-red handle without a scene-space ROI.

    Ambiguous similarly-red objects cause a perception failure rather than a
    guess. The RGB predicate is a shared asset appearance prior; it is not an
    object-coordinate oracle and never reads segmentation truth from MuJoCo.
    """
    color = np.asarray(rgb, dtype=np.float32)
    if color.shape != (480, 640, 3):
        raise KnifePerceptionError("public RGB must be real 640x480")
    red = ((color[:, :, 0] > 60) &
           (color[:, :, 0] > 2.5 * color[:, :, 1]) &
           (color[:, :, 0] > 2.5 * color[:, :, 2]) &
           (color[:, :, 1] < 70) & (color[:, :, 2] < 70))
    components, count = label(red)
    sizes = np.bincount(components.ravel(), minlength=count + 1)[1:]
    if not len(sizes):
        raise KnifePerceptionError("no dark-red handle visible")
    order = np.argsort(sizes)[::-1]
    largest = int(sizes[order[0]])
    second = int(sizes[order[1]]) if len(order) > 1 else 0
    if largest < 40:
        raise KnifePerceptionError("red handle is too small or occluded")
    if second >= max(25, int(.45 * largest)):
        raise KnifePerceptionError("multiple plausible red handles visible")
    return components == int(order[0] + 1)


def knife_box_from_public_camera(camera: dict):
    """Fallback handle box from the same public RGB-D when SAM misses it.

    The box names the operated knife for the action gate. It does not reveal
    a grasp site, simulator object ID, dose/hand state, or evaluator truth.
    Multiple plausible red components fail closed in _public_handle_mask.
    """
    mask = _public_handle_mask(np.asarray(camera["rgb"]))
    box = world_box_from_mask(
        name="knife", mask=mask,
        depth_m=np.asarray(camera["depth_m"], dtype=float),
        intrinsic_k=np.asarray(camera["intrinsics"], dtype=float),
        extrinsic_cam2world=np.asarray(camera["extrinsics"], dtype=float),
        margin_m=.03)
    return box, {
        "name": "knife", "prompt": "visible dark-red knife handle",
        "score": None, "mask_pixels": int(mask.sum()),
        "box_lower": box.lower, "box_upper": box.upper,
        "source": "public_rgbd_perception",
        "backend": "deterministic_public_red_handle_fallback",
    }


def estimate_knife_handle(packet: dict, query: str) -> KnifePickEstimate:
    if not isinstance(query, str) or not 0 < len(query.strip()) <= 100:
        raise KnifePerceptionError("query must name the visible knife")
    if not any(word in query.lower() for word in ("knife", "red handle")):
        raise KnifePerceptionError("this E1 candidate only supports the knife")
    cameras = packet.get("cameras")
    if not isinstance(cameras, dict) or set(cameras) != {"agentview", "robot0_eye_in_hand"}:
        raise KnifePerceptionError("both declared public cameras are required")
    view, wrist = cameras["agentview"], cameras["robot0_eye_in_hand"]
    frame = view.get("frame_id")
    if not isinstance(frame, int) or frame < 0 or wrist.get("frame_id") != frame:
        raise KnifePerceptionError("public camera frames disagree")
    for camera in (view, wrist):
        rgb = np.asarray(camera.get("rgb"))
        depth = np.asarray(camera.get("depth_m"), dtype=float)
        k = np.asarray(camera.get("intrinsics"), dtype=float)
        e = np.asarray(camera.get("extrinsics"), dtype=float)
        if (rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8 or
                depth.shape != (480, 640) or not np.isfinite(depth).all() or
                np.any(depth <= 0) or k.shape != (3, 3) or e.shape != (4, 4) or
                not np.isfinite(k).all() or not np.isfinite(e).all() or
                abs(np.linalg.det(k)) < 1e-9):
            raise KnifePerceptionError("public RGB-D or calibration invalid")
    mask = _public_handle_mask(view["rgb"])
    depth = np.asarray(view["depth_m"], dtype=float)
    ys, xs = np.nonzero(mask & (depth < 3.0))
    if len(xs) < 40:
        raise KnifePerceptionError("too few valid public handle-depth pixels")
    z = depth[ys, xs]
    k = np.asarray(view["intrinsics"], dtype=float)
    e = np.asarray(view["extrinsics"], dtype=float)
    cam = np.linalg.inv(k) @ np.vstack((xs * z, ys * z, z))
    world = (e @ np.vstack((cam, np.ones_like(z))))[:3].T
    if not np.isfinite(world).all():
        raise KnifePerceptionError("handle cloud projection invalid")
    lo, hi = np.quantile(world, [.05, .95], axis=0)
    span = hi - lo
    if not (.012 <= span[0] <= .10 and .005 <= span[1] <= .07 and
            .001 <= span[2] <= .045):
        raise KnifePerceptionError("visible handle dimensions implausible")
    # Grasp near the blade end of the *visible* handle, at gripper-center
    # height above its visible top. Both are generic knife/tool priors.
    target = (float(np.quantile(world[:, 0], .80)),
              float(np.median(world[:, 1])),
              float(np.quantile(world[:, 2], .90) + .014))
    if not (-.5 < target[0] < .5 and -.5 < target[1] < .5 and .90 < target[2] < 1.1):
        raise KnifePerceptionError("estimated handle outside tabletop workspace")
    return KnifePickEstimate(target, frame, len(xs), tuple(map(float, span)))


class KnifePickToolkit:
    """Outer macro wrapper: `base` is the common budgeted, gated toolkit."""

    def __init__(self, base, *, public_packet: Callable[[], dict],
                 log_event: Callable[[dict], None] | None = None):
        self.base = base
        self.public_packet = public_packet
        self.log_event = log_event
        self.events: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_tools_spec(self):
        specs = self.base.get_tools_spec()
        if any(spec["name"] == "pick" for spec in specs):
            raise RuntimeError("pick already registered")
        return [*specs, PICK_SPEC]

    def _log(self, event: dict):
        self.events.append(event)
        if self.log_event is not None:
            self.log_event(event)

    def _fail(self, code: str, reason: str, children: list[dict],
              estimate: KnifePickEstimate | None = None):
        self._log({"event": "knife_pick_failed", "code": code, "reason": reason,
                   "child_calls": len(children)})
        return _tool_result("pick", {
            "success": False, "code": code, "error": reason,
            "estimate": asdict(estimate) if estimate else None,
            "child_calls": children,
        })

    def execute_tool(self, name: str, input_dict: dict[str, Any]):
        if name != "pick":
            return self.base.execute_tool(name, input_dict)
        children: list[dict] = []
        if not isinstance(input_dict, dict) or set(input_dict) != {"query"}:
            return self._fail("invalid_arguments", "pick requires only query", children)
        if min(self.base.per_round_budget - self.base.round_calls,
               self.base.episode_budget - self.base.episode_calls) < 6:
            return self._fail("budget_insufficient", "six child primitive calls require budget", children)
        try:
            packet = self.public_packet()
            if packet["robot_state"].get("grasp_latched") is not False:
                raise KnifePerceptionError("public gripper latch must be false")
            estimate = estimate_knife_handle(packet, input_dict["query"])
        except Exception as exc:
            return self._fail("perception_failed", f"{type(exc).__name__}: {exc}", children)
        self._log({"event": "knife_pick_estimate", "query": input_dict["query"],
                   "estimate": asdict(estimate)})
        x, y, z = estimate.xyz_m
        moves = (
            ("stage_behind", "move_to", {"xyz": [-.25, y, 1.18], "gripper": -1,
                                         "max_steps": 180, "tol": .012, "step_clip": .012}),
            ("above_handle", "move_to", {"xyz": [x, y, 1.18], "gripper": -1,
                                         "max_steps": 180, "tol": .010, "step_clip": .012}),
            ("approach_handle", "move_to", {"xyz": [x, y, 1.035], "gripper": -1,
                                            "max_steps": 180, "tol": .008, "step_clip": .007}),
            ("lower_to_handle", "move_to", {"xyz": [x, y, z], "gripper": -1,
                                            "max_steps": 120, "tol": .005, "step_clip": .005}),
            ("close", "set_gripper", {"gripper": 1, "steps": 26}),
            ("lift", "move_to", {"xyz": [x, y, 1.09], "gripper": 1,
                                  "max_steps": 180, "tol": .010, "step_clip": .006}),
        )
        closed_z: float | None = None
        # The outer public pick(query) is already a declared operated-object
        # intent. Preserve that identity for each low-level child review,
        # without adding a planner tool parameter or bypassing any gate.
        scope_factory = getattr(self.base, "operation_scope", None)
        scope = scope_factory("knife", input_dict["query"]) if callable(scope_factory) else nullcontext()
        with scope:
            for stage, tool, args in moves:
                before = self.base.episode_calls
                try:
                    result = self.base.execute_tool(tool, args)
                    public = result.result if isinstance(result.result, dict) else {}
                    after = self.public_packet()
                    latch = after["robot_state"].get("grasp_latched")
                    eef = np.asarray(after["robot_state"].get("eef_pos"), dtype=float)
                except Exception as exc:
                    return self._fail("child_exception", f"{stage}: {type(exc).__name__}: {exc}",
                                      children, estimate)
                child = {"stage": stage, "tool": tool, "arguments": args,
                         "budgeted_calls_delta": self.base.episode_calls - before,
                         "public_frame_id": after.get("frame_id"),
                         "public_eef_xyz_m": eef.tolist(), "public_grasp_latched": latch,
                         "code": public.get("code"), "error": public.get("error")}
                children.append(child)
                self._log({"event": "knife_pick_child", **child})
                if public.get("error"):
                    return self._fail(str(public.get("code") or "child_failed"),
                                      str(public["error"]), children, estimate)
                if child["budgeted_calls_delta"] != 1 or eef.shape != (3,) or not np.isfinite(eef).all():
                    return self._fail("invalid_child_result", "child budget or public state invalid",
                                      children, estimate)
                if stage not in {"close", "lift"} and np.linalg.norm(eef - np.asarray(args["xyz"])) > .025:
                    return self._fail("motion_incomplete", f"{stage} did not reach target",
                                      children, estimate)
                if stage == "close":
                    if latch is not True:
                        return self._fail("grasp_not_attached", "public latch stayed false", children, estimate)
                    closed_z = float(eef[2])
                if stage == "lift" and (latch is not True or closed_z is None or
                                        float(eef[2]) - closed_z < .020):
                    return self._fail("lift_incomplete", "public latch/lift evidence missing",
                                      children, estimate)
        self._log({"event": "knife_pick_complete", "child_calls": len(children)})
        return _tool_result("pick", {
            "success": True, "estimate": asdict(estimate),
            "child_calls": children, "public_grasp_latched": True,
            "note": "Only a local lift is verified online; offline scoring must verify object lift and event.",
        })
