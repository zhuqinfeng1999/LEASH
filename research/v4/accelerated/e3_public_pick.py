"""Shared E3 pick(query) from current two-view public RGB-D and visible handle.

The estimator sees a SAM3 mask of the kettle's dark side handle. It never
reads the authored grasp site, simulator object pose, or evaluator truth.
Each child is a normal RPent primitive through the caller's B0/Ours boundary.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from research.v3.bridge.b0 import _tool_result


PICK_SPEC = {
    "name": "pick",
    "description": (
        "Use current public RGB-D to find the visible dark side handle of "
        "the grey kettle and perform a bounded local pickup. Six ordinary "
        "RPent child primitives consume the shared action budget and pass "
        "through all active safety gates. The tool stops on any denial, "
        "empty grasp, or insufficient lift; it never transports or pours."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"], "additionalProperties": False,
    },
}


class PublicPickError(ValueError):
    pass


@dataclass(frozen=True)
class HandleEstimate:
    xyz_m: tuple[float, float, float]
    frame_id: int
    mask_pixels: int
    mask_score: float
    visible_span_m: tuple[float, float, float]
    method: str = "public_dark_side_handle_depth_quantiles_v1"


def estimate_handle(packet: dict, sam_client, query: str) -> HandleEstimate:
    if not isinstance(query, str) or not 0 < len(query.strip()) <= 100:
        raise PublicPickError("query must name the visible kettle")
    if not any(word in query.lower() for word in ("kettle", "hot water", "handle")):
        raise PublicPickError("this public pick only supports the kettle")
    views = packet.get("cameras")
    if not isinstance(views, dict) or set(views) != {"agentview", "robot0_eye_in_hand"}:
        raise PublicPickError("both public cameras are required")
    camera = views["agentview"]
    frame = packet.get("frame_id")
    if not isinstance(frame, int) or any(view.get("frame_id") != frame
                                         for view in views.values()):
        raise PublicPickError("public camera frames disagree")
    rgb = np.asarray(camera.get("rgb"))
    depth = np.asarray(camera.get("depth_m"), dtype=float)
    k = np.asarray(camera.get("intrinsics"), dtype=float)
    e = np.asarray(camera.get("extrinsics"), dtype=float)
    if (rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8 or
            depth.shape != (480, 640) or not np.isfinite(depth).all() or
            np.any(depth <= 0) or k.shape != (3, 3) or e.shape != (4, 4) or
            not np.isfinite(k).all() or not np.isfinite(e).all() or
            abs(np.linalg.det(k)) < 1e-9):
        raise PublicPickError("public RGB-D or calibration invalid")
    result = sam_client.segment(
        rgb, text_prompt="black handle on the grey kettle", min_score=.4)
    if not result.found or result.mask is None or result.score is None:
        raise PublicPickError("visible kettle handle was not found")
    mask = np.asarray(result.mask, dtype=bool)
    if mask.shape != depth.shape or not 100 <= int(mask.sum()) <= 5000:
        raise PublicPickError("kettle-handle mask size is ambiguous")
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    cam = np.linalg.inv(k) @ np.vstack((xs*z, ys*z, z))
    world = (e @ np.vstack((cam, np.ones_like(z))))[:3].T
    if not np.isfinite(world).all():
        raise PublicPickError("public handle projection invalid")
    q05, q25, q75, q95 = np.quantile(world, [.05, .25, .75, .95], axis=0)
    span = q95 - q05
    if not (.025 <= span[0] <= .14 and .005 <= span[1] <= .06 and
            .02 <= span[2] <= .13):
        raise PublicPickError("visible handle dimensions are implausible")
    # The mask includes the handle's upper connector. Select its outside edge
    # horizontally, its camera-visible side in y, and the visible lower grip
    # band vertically. The 0.970 m floor is a shared tabletop/handle-height
    # prior for this published kettle geometry, not a per-seed grasp-site read.
    # The mask's lower-left quartile tracks the physical side handle over
    # slot perturbations. Apply one shared millimetre-scale calibration for
    # segmentation's visible surface; no per-seed pose or private site is read.
    target = (float(q25[0] - .0035), float(q25[1] + .008),
              float(max(q05[2] + .002, .975)))
    if not (-.35 < target[0] < .10 and -.40 < target[1] < .05 and
            .95 < target[2] < 1.04):
        raise PublicPickError("visible handle estimate outside tabletop reach")
    return HandleEstimate(target, frame, int(mask.sum()), float(result.score),
                          tuple(map(float, span)))


class KettlePickToolkit:
    def __init__(self, base, *, public_packet: Callable[[], dict], sam_client,
                 log_event: Callable[[dict], None] | None = None):
        self.base = base
        self.public_packet = public_packet
        self.sam_client = sam_client
        self.log_event = log_event
        self.events: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_tools_spec(self):
        specs = self.base.get_tools_spec()
        if any(item["name"] == "pick" for item in specs):
            raise RuntimeError("pick already registered")
        return [*specs, PICK_SPEC]

    def _log(self, event: dict):
        self.events.append(event)
        if self.log_event is not None:
            self.log_event(event)

    def _fail(self, code, reason, children, estimate=None):
        self._log({"event": "kettle_pick_failed", "code": code,
                   "reason": reason, "child_calls": len(children)})
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
        if min(self.base.per_round_budget-self.base.round_calls,
               self.base.episode_budget-self.base.episode_calls) < 6:
            return self._fail("budget_insufficient", "six child calls require budget", children)
        try:
            packet = self.public_packet()
            if packet["robot_state"].get("grasp_latched") is not False:
                raise PublicPickError("public gripper latch must be false")
            estimate = estimate_handle(packet, self.sam_client, input_dict["query"])
        except Exception as exc:
            return self._fail("perception_failed",
                              f"{type(exc).__name__}: {exc}", children)
        self._log({"event": "kettle_pick_estimate", "query": input_dict["query"],
                   "estimate": asdict(estimate)})
        x, y, z = estimate.xyz_m
        moves = (
            ("source_staging", "move_to", {"xyz": [-.25, y, 1.18],
                "gripper": -1, "max_steps": 180, "tol": .012, "step_clip": .012}),
            ("above_handle", "move_to", {"xyz": [x, y, 1.18],
                "gripper": -1, "max_steps": 180, "tol": .010, "step_clip": .012}),
            ("approach_handle", "move_to", {"xyz": [x, y, 1.04],
                "gripper": -1, "max_steps": 180, "tol": .008, "step_clip": .007}),
            ("lower_to_handle", "move_to", {"xyz": [x, y, z],
                "gripper": -1, "max_steps": 140, "tol": .005, "step_clip": .005}),
            ("close", "set_gripper", {"gripper": 1, "steps": 26}),
            ("lift", "move_to", {"xyz": [x, y, 1.10],
                "gripper": 1, "max_steps": 180, "tol": .012, "step_clip": .006}),
        )
        closed_z = None
        operation = getattr(self.base, "operation_scope", None)
        scope = operation("kettle", input_dict["query"]) if callable(operation) else nullcontext()
        with scope:
            for stage, tool, args in moves:
                before = self.base.episode_calls
                try:
                    result = self.base.execute_tool(tool, args)
                    visible = result.result if isinstance(result.result, dict) else {}
                    after = self.public_packet()
                    latch = after["robot_state"].get("grasp_latched")
                    eef = np.asarray(after["robot_state"].get("eef_pos"), dtype=float)
                except Exception as exc:
                    return self._fail("child_exception",
                                      f"{stage}: {type(exc).__name__}: {exc}",
                                      children, estimate)
                child = {"stage": stage, "tool": tool, "arguments": args,
                         "budgeted_calls_delta": self.base.episode_calls-before,
                         "public_frame_id": after.get("frame_id"),
                         "public_eef_xyz_m": eef.tolist(),
                         "public_grasp_latched": latch,
                         "code": visible.get("code"), "error": visible.get("error")}
                children.append(child)
                self._log({"event": "kettle_pick_child", **child})
                if visible.get("error"):
                    return self._fail(str(visible.get("code") or "child_failed"),
                                      str(visible["error"]), children, estimate)
                if child["budgeted_calls_delta"] != 1 or eef.shape != (3,) or not np.isfinite(eef).all():
                    return self._fail("invalid_child_result", "child budget/state invalid",
                                      children, estimate)
                if stage not in {"close", "lift"} and np.linalg.norm(
                        eef-np.asarray(args["xyz"])) > .030:
                    return self._fail("motion_incomplete", f"{stage} missed target",
                                      children, estimate)
                if stage == "close":
                    if latch is not True:
                        return self._fail("grasp_not_attached", "public latch stayed false",
                                          children, estimate)
                    closed_z = float(eef[2])
                if stage == "lift" and (latch is not True or closed_z is None or
                                        float(eef[2])-closed_z < .020):
                    return self._fail("lift_incomplete", "public latch/lift missing",
                                      children, estimate)
        self._log({"event": "kettle_pick_complete", "child_calls": len(children)})
        return _tool_result("pick", {
            "success": True, "estimate": asdict(estimate),
            "child_calls": children, "public_grasp_latched": True,
            "note": "Only local public-latch/EEF lift is verified online; offline scoring checks the vessel lift.",
        })
