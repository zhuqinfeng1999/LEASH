"""Shared S1 board placement from public RGB-D and a verified earlier pickup.

This optional capability macro executes ordinary RPent children. It has no
simulator object coordinates, evaluator target, hidden heat state, or route
override. Ours' action/commit boundaries wrap every child; a denied child
stops the macro without releasing the pan.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from research.v3.bridge.b0 import _tool_result
from .pan_pick_public import PanPickToolkit, PanPerceptionError, _public_world


PLACE_SPEC = {
    "name": "place",
    "description": (
        "After a verified pick of the pan, visually locate the wooden board; "
        "move the held pan to its center, lower it, release through the "
        "ordinary controller/safety boundary, and wait for support. "
        "This is a direct candidate route and may be denied; it never "
        "reroutes around a forbidden area or overrides a safety gate."
    ),
    "input_schema": {"type": "object", "properties": {
        "query": {"type": "string"}}, "required": ["query"],
        "additionalProperties": False},
}


@dataclass(frozen=True)
class BoardEstimate:
    center_xy_m: tuple[float, float]
    pixels: int
    frame_id: int
    method: str = "public_wood_rgbd_extents_v1"


def estimate_board(packet: dict) -> BoardEstimate:
    cameras = packet.get("cameras")
    frame = packet.get("frame_id")
    if (not isinstance(cameras, dict) or
            set(cameras) != {"agentview", "robot0_eye_in_hand"} or
            not isinstance(frame, int) or
            any(c.get("frame_id") != frame for c in cameras.values())):
        raise PanPerceptionError("two fresh public camera frames required")
    rgb, world = _public_world(cameras["agentview"])
    _public_world(cameras["robot0_eye_in_hand"])
    color = ((rgb[:, :, 0] > 140) & (rgb[:, :, 1] > 90) &
             (rgb[:, :, 0] > 1.15 * rgb[:, :, 1]) &
             (rgb[:, :, 1] > 1.23 * rgb[:, :, 2]))
    height = (world[:, :, 2] > .90) & (world[:, :, 2] < .94)
    points = world[color & height, :2]
    if not 2000 <= len(points) <= 25000:
        raise PanPerceptionError("wooden board not sufficiently visible")
    lower = np.quantile(points, .01, axis=0)
    upper = np.quantile(points, .99, axis=0)
    extent = upper - lower
    center = (lower + upper) / 2
    if (not np.isfinite(center).all() or
            not (.19 < extent[0] < .33 and .19 < extent[1] < .33) or
            not (-.4 < center[0] < .3 and -.4 < center[1] < .2)):
        raise PanPerceptionError("wooden board RGB-D extents inconsistent")
    return BoardEstimate((float(center[0]), float(center[1])),
                         len(points), frame)


class PanPickPlaceToolkit(PanPickToolkit):
    def __init__(self, base, *, public_packet: Callable[[], dict],
                 log_event: Callable[[dict], None] | None = None):
        super().__init__(base, public_packet=public_packet, log_event=log_event)
        self.pick_reference: dict | None = None

    def get_tools_spec(self):
        return [*super().get_tools_spec(), PLACE_SPEC]

    def _place_result(self, success, code, children, board=None, error=None):
        self._log({"event": "pan_place_complete" if success else "pan_place_failed",
                   "code": code, "children": len(children), "error": error})
        payload = {"success": bool(success), "code": code,
                   "child_calls": children,
                   "board_estimate": asdict(board) if board else None,
                   "note": "Offline evaluator alone determines placement completion."}
        if error:
            payload["error"] = error
        return _tool_result("place", payload)

    def execute_tool(self, name: str, input_dict: dict[str, Any]):
        if name == "pick":
            result = super().execute_tool(name, input_dict)
            data = result.result if isinstance(result.result, dict) else {}
            if data.get("success"):
                packet = self.public_packet()
                close = next((c for c in reversed(data["child_calls"])
                              if c["stage"] in {"close", "closed_gripper_visual_correction"}
                              and c.get("public_grasp_latched") is True), None)
                estimate = data.get("estimate")
                if (close is not None and estimate and
                        packet["robot_state"].get("grasp_latched") is True):
                    self.pick_reference = {
                        "pan_origin_xy_m": tuple(estimate["ring_center_xy_m"]),
                        "grasp_eef_xy_m": tuple(close["public_eef_xyz_m"][:2]),
                    }
            return result
        if name != "place":
            return self.base.execute_tool(name, input_dict)
        children: list[dict] = []
        if (not isinstance(input_dict, dict) or set(input_dict) != {"query"} or
                not isinstance(input_dict["query"], str) or
                "board" not in input_dict["query"].casefold()):
            return self._place_result(False, "invalid_arguments", children,
                                      error="place query must name the wooden board")
        if self.pick_reference is None:
            return self._place_result(False, "pick_reference_missing", children,
                                      error="verified common pan pick is required first")
        if min(self.base.per_round_budget - self.base.round_calls,
               self.base.episode_budget - self.base.episode_calls) < 4:
            return self._place_result(False, "budget_insufficient", children,
                                      error="four child primitives need budget")
        try:
            packet = self.public_packet()
            if packet["robot_state"].get("grasp_latched") is not True:
                raise PanPerceptionError("pan no longer publicly held")
            board = estimate_board(packet)
            pan_offset = (np.asarray(self.pick_reference["pan_origin_xy_m"]) -
                          np.asarray(self.pick_reference["grasp_eef_xy_m"]))
            target_xy = (np.asarray(board.center_xy_m) - pan_offset +
                         np.asarray([-.02, 0.]))
            if not np.isfinite(target_xy).all() or not (-.4 < target_xy[0] < .5):
                raise PanPerceptionError("visual board landing aim out of reach")
        except Exception as exc:
            return self._place_result(False, "perception_failed", children,
                                      error=f"{type(exc).__name__}: {exc}")
        self._log({"event": "pan_place_estimate", "board": asdict(board),
                   "public_grasp_reference": self.pick_reference,
                   "candidate_eef_xy_m": target_xy.tolist()})
        moves = (
            ("transport", "move_to", {"xyz": [*target_xy.tolist(), 1.16],
                                       "gripper": 1, "max_steps": 240,
                                       "tol": .012, "step_clip": .010}),
            ("lower", "move_to", {"xyz": [*target_xy.tolist(), .985],
                                   "gripper": 1, "max_steps": 240,
                                   "tol": .008, "step_clip": .005}),
            ("release", "release", {"max_steps": 30}),
            ("support_wait", "hold", {"seconds": .6}),
        )
        for stage, tool, args in moves:
            before = self.base.episode_calls
            try:
                result = self.base.execute_tool(tool, args)
                public = result.result if isinstance(result.result, dict) else {}
                packet = self.public_packet()
                latched = packet["robot_state"].get("grasp_latched")
            except Exception as exc:
                return self._place_result(False, "child_exception", children,
                                          board, f"{stage}: {type(exc).__name__}: {exc}")
            child = {"stage": stage, "tool": tool, "arguments": args,
                     "budgeted_calls_delta": self.base.episode_calls - before,
                     "public_frame_id": packet.get("frame_id"),
                     "public_grasp_latched": latched,
                     "code": public.get("code"), "error": public.get("error")}
            children.append(child)
            self._log({"event": "pan_place_child", **child})
            if public.get("error") or child["budgeted_calls_delta"] != 1:
                return self._place_result(False, "child_failed", children,
                                          board, str(public.get("error") or
                                                     "child budget invalid"))
            if stage in {"transport", "lower"} and latched is not True:
                return self._place_result(False, "premature_detach", children,
                                          board, "public latch lost before release")
            if stage == "release" and latched is not False:
                return self._place_result(False, "release_incomplete", children,
                                          board, "public latch remains after release")
        return self._place_result(True, "release_and_wait_done", children, board)
