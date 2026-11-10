"""Shared S1 pan-handle pickup from the two public RGB-D cameras.

The red hob ring and dark physical handle are visible asset features. The
estimator never reads the simulator pan pose, grasp site, evaluator, or weld
metadata. The macro sends ordinary budgeted RPent child tools through the
same B0/Ours execution boundary; it does not transport or release the pan.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares

from research.v3.bridge.b0 import _tool_result


PICK_SPEC = {
    "name": "pick",
    "description": (
        "Locate the pan's visible dark handle using the red burner ring and "
        "current public RGB-D. Perform up to seven bounded, budgeted RPent child "
        "actions to grasp and lift it. Every child passes the same safety "
        "gates. This tool does not move the pan to the board or release it."
    ),
    "input_schema": {"type": "object", "properties": {
        "query": {"type": "string"}}, "required": ["query"],
        "additionalProperties": False},
}


class PanPerceptionError(ValueError):
    pass


@dataclass(frozen=True)
class PanHandleEstimate:
    xyz_m: tuple[float, float, float]
    frame_id: int
    ring_center_xy_m: tuple[float, float]
    ring_pixels: int
    handle_pixels: int
    fit_p95_m: float
    method: str = "public_red_ring_dark_handle_rgbd_v1"


def _public_world(camera: dict) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(camera.get("rgb"))
    depth = np.asarray(camera.get("depth_m"), dtype=float)
    k = np.asarray(camera.get("intrinsics"), dtype=float)
    e = np.asarray(camera.get("extrinsics"), dtype=float)
    if (rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8 or
            depth.shape != (480, 640) or not np.isfinite(depth).all() or
            np.any(depth <= 0) or k.shape != (3, 3) or e.shape != (4, 4) or
            not np.isfinite(k).all() or not np.isfinite(e).all() or
            abs(np.linalg.det(k)) < 1e-9):
        raise PanPerceptionError("fresh calibrated 640x480 public RGB-D required")
    yy, xx = np.indices(depth.shape)
    z = depth.ravel()
    world = (e @ np.vstack((np.linalg.inv(k) @ np.vstack(
        (xx.ravel() * z, yy.ravel() * z, z)), np.ones_like(z))))[:3].T
    return rgb, world.reshape(480, 640, 3)


def estimate_pan_handle(packet: dict, query: str) -> PanHandleEstimate:
    if (not isinstance(query, str) or not 0 < len(query.strip()) <= 100 or
            not any(word in query.lower() for word in ("pan", "skillet", "handle"))):
        raise PanPerceptionError("query must name the visible pan or handle")
    cameras = packet.get("cameras")
    if not isinstance(cameras, dict) or set(cameras) != {"agentview", "robot0_eye_in_hand"}:
        raise PanPerceptionError("both declared public cameras required")
    frame = packet.get("frame_id")
    if (not isinstance(frame, int) or frame < 0 or
            any(camera.get("frame_id") != frame for camera in cameras.values())):
        raise PanPerceptionError("public camera frames disagree")
    robot = packet.get("robot_state", {})
    if robot.get("grasp_latched") is not False:
        raise PanPerceptionError("pickup requires a publicly empty gripper")
    rgb, world = _public_world(cameras["agentview"])
    _public_world(cameras["robot0_eye_in_hand"])  # freshness/calibration check
    # Red ring lies on the work surface; height rejects red clothing and labels.
    red = ((rgb[:, :, 0] > 120) &
           (rgb[:, :, 0] > 1.8 * rgb[:, :, 1]) &
           (rgb[:, :, 0] > 1.8 * rgb[:, :, 2]) &
           (rgb[:, :, 1] < 100) & (rgb[:, :, 2] < 100) &
           (world[:, :, 2] > .905) & (world[:, :, 2] < .93))
    ring = world[red, :2]
    if not 120 <= len(ring) <= 2500:
        raise PanPerceptionError("red hob ring is missing or ambiguous")
    # The displayed appliance has a declared 118-mm ring radius. Fit its
    # visible arc in metric public depth; occlusion makes its pixel centroid
    # an unsafe estimate of the center.
    residual = lambda center: np.linalg.norm(ring - center, axis=1) - .118
    fit = least_squares(residual, np.median(ring, axis=0), max_nfev=100)
    center = np.asarray(fit.x, dtype=float)
    p95 = float(np.quantile(np.abs(residual(center)), .95))
    if (not fit.success or not np.isfinite(center).all() or p95 > .012 or
            not (-.5 < center[0] < .5 and -.5 < center[1] < .5)):
        raise PanPerceptionError("public red-ring circle fit is inconsistent")
    # Restrict to the dark, tabletop-height handle outside the fitted ring.
    # Median visible surface is the grasp aim; 8 mm is a generic gripper-pad
    # center offset, not a hidden simulator grasp-site coordinate.
    handle = ((world[:, :, 0] > center[0] + .13) &
              (world[:, :, 0] < center[0] + .25) &
              (np.abs(world[:, :, 1] - center[1]) < .07) &
              (world[:, :, 2] > .92) & (world[:, :, 2] < .98) &
              (rgb.max(axis=2) < 95))
    points = world[handle]
    if not 70 <= len(points) <= 1500:
        raise PanPerceptionError("dark pan handle is missing or ambiguous")
    aim = np.median(points, axis=0)
    aim[2] -= .008
    if not (-.5 < aim[0] < .5 and -.5 < aim[1] < .5 and .93 < aim[2] < 1.02):
        raise PanPerceptionError("public pan handle aim outside qualified tabletop")
    return PanHandleEstimate(tuple(float(v) for v in aim), frame,
                             tuple(float(v) for v in center), len(ring),
                             len(points), p95)


class PanPickToolkit:
    """Expose one optional macro equally to B0 and Ours."""

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

    def _log(self, event):
        self.events.append(event)
        if self.log_event is not None:
            self.log_event(event)

    def _result(self, *, success: bool, code: str, children, estimate=None,
                reason: str | None = None):
        payload = {"success": success, "code": code,
                   "child_calls": children,
                   "estimate": asdict(estimate) if estimate else None,
                   "note": "Offline evaluator determines task completion."}
        if reason is not None:
            payload["error"] = reason
        self._log({"event": "pan_pick_complete" if success else "pan_pick_failed",
                   "code": code, "children": len(children), "reason": reason})
        return _tool_result("pick", payload)

    def execute_tool(self, name: str, input_dict: dict[str, Any]):
        if name != "pick":
            return self.base.execute_tool(name, input_dict)
        children = []
        if not isinstance(input_dict, dict) or set(input_dict) != {"query"}:
            return self._result(success=False, code="invalid_arguments",
                                children=children, reason="pick requires only query")
        if min(self.base.per_round_budget - self.base.round_calls,
               self.base.episode_budget - self.base.episode_calls) < 7:
            return self._result(success=False, code="budget_insufficient",
                                children=children, reason="seven child primitives need budget")
        try:
            estimate = estimate_pan_handle(self.public_packet(), input_dict["query"])
        except Exception as exc:
            return self._result(success=False, code="perception_failed", children=children,
                                reason=f"{type(exc).__name__}: {exc}")
        self._log({"event": "pan_pick_estimate", "estimate": asdict(estimate)})
        x, y, z = estimate.xyz_m
        moves = (
            ("stage_behind", "move_to", {"xyz": [-.25, y, 1.18], "gripper": -1,
                                          "max_steps": 180, "tol": .012, "step_clip": .012}),
            ("above_handle", "move_to", {"xyz": [x, y, 1.18], "gripper": -1,
                                           "max_steps": 180, "tol": .010, "step_clip": .012}),
            ("approach_handle", "move_to", {"xyz": [x, y, 1.03], "gripper": -1,
                                              "max_steps": 180, "tol": .008, "step_clip": .007}),
            ("lower_to_handle", "move_to", {"xyz": [x, y, z], "gripper": -1,
                                              "max_steps": 140, "tol": .005, "step_clip": .005}),
            # Close while approaching along the visible handle. Closing in
            # place first pushes this thin handle away before both pads can
            # bracket it; the coupled command keeps finger motion visible.
            ("close", "move_to", {"xyz": [x + .035, y, z], "gripper": 1,
                                    "max_steps": 120, "tol": .004,
                                    "step_clip": .003}),
            ("lift", "move_to", {"xyz": [x, y, 1.12], "gripper": 1,
                                  "max_steps": 180, "tol": .010, "step_clip": .006}),
        )
        close_z = None
        for stage, tool, args in moves:
            before = self.base.episode_calls
            try:
                result = self.base.execute_tool(tool, args)
                public = result.result if isinstance(result.result, dict) else {}
                packet = self.public_packet()
                latched = packet["robot_state"].get("grasp_latched")
                eef = np.asarray(packet["robot_state"].get("eef_pos"), dtype=float)
            except Exception as exc:
                return self._result(success=False, code="child_exception", children=children,
                                    estimate=estimate,
                                    reason=f"{stage}: {type(exc).__name__}: {exc}")
            child = {"stage": stage, "tool": tool, "arguments": args,
                     "budgeted_calls_delta": self.base.episode_calls - before,
                     "public_frame_id": packet.get("frame_id"),
                     "public_eef_xyz_m": eef.tolist(), "public_grasp_latched": latched,
                     "code": public.get("code"), "error": public.get("error")}
            children.append(child)
            self._log({"event": "pan_pick_child", **child})
            if public.get("error"):
                return self._result(success=False, code=str(public.get("code") or "child_failed"),
                                    children=children, estimate=estimate,
                                    reason=str(public["error"]))
            if (child["budgeted_calls_delta"] != 1 or eef.shape != (3,) or
                    not np.isfinite(eef).all()):
                return self._result(success=False, code="invalid_child_result",
                                    children=children, estimate=estimate,
                                    reason="child budget or public state invalid")
            if stage not in {"close", "lift"} and np.linalg.norm(eef - args["xyz"]) > .025:
                return self._result(success=False, code="motion_incomplete",
                                    children=children, estimate=estimate,
                                    reason=f"{stage} did not reach target")
            if stage == "close":
                if latched is not True:
                    # Initial finger contact can slide the pan. Reobserve its
                    # visible handle, then perform one small closed-gripper
                    # tracking move. This child still passes the ordinary
                    # action/commit boundary and costs one primitive call.
                    try:
                        fresh = estimate_pan_handle(self.public_packet(), input_dict["query"])
                        fx, fy, fz = fresh.xyz_m
                        correction = {"xyz": [fx + .025, fy, fz], "gripper": 1,
                                      "max_steps": 100, "tol": .004,
                                      "step_clip": .003}
                        before = self.base.episode_calls
                        correction_result = self.base.execute_tool("move_to", correction)
                        correction_public = (correction_result.result
                                             if isinstance(correction_result.result, dict) else {})
                        after = self.public_packet()
                        latched = after["robot_state"].get("grasp_latched")
                        eef = np.asarray(after["robot_state"].get("eef_pos"), dtype=float)
                    except Exception as exc:
                        return self._result(success=False, code="tracking_failed",
                                            children=children, estimate=estimate,
                                            reason=f"closed-gripper visual correction: {type(exc).__name__}: {exc}")
                    correction_child = {
                        "stage": "closed_gripper_visual_correction", "tool": "move_to",
                        "arguments": correction,
                        "budgeted_calls_delta": self.base.episode_calls - before,
                        "public_frame_id": after.get("frame_id"),
                        "public_eef_xyz_m": eef.tolist(), "public_grasp_latched": latched,
                        "code": correction_public.get("code"),
                        "error": correction_public.get("error"),
                        "fresh_public_estimate": asdict(fresh),
                    }
                    children.append(correction_child)
                    self._log({"event": "pan_pick_child", **correction_child})
                    if correction_public.get("error") or correction_child["budgeted_calls_delta"] != 1:
                        return self._result(success=False, code="tracking_child_failed",
                                            children=children, estimate=estimate,
                                            reason=str(correction_public.get("error") or
                                                       "tracking child budget invalid"))
                    if latched is not True:
                        return self._result(success=False, code="grasp_not_attached",
                                            children=children, estimate=estimate,
                                            reason="public latch stayed false after one visual correction")
                close_z = float(eef[2])
            if stage == "lift" and (latched is not True or close_z is None or
                                    float(eef[2]) - close_z < .02):
                return self._result(success=False, code="lift_incomplete",
                                    children=children, estimate=estimate,
                                    reason="public latch/lift evidence missing")
        return self._result(success=True, code="lift_verified", children=children,
                            estimate=estimate)
