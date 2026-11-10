"""Finite P1-03 preclose-only check on the current common-control stack.

This is a no-model engineering diagnostic.  It replays the exact seed-0
preclose prefix saved by the current P1 single-right-arm layout, routes its
two ordinary translations through the current common fixed-rotation control,
and leaves the pose descent on the original RPent primitive.  It never sends
CLOSE and never lifts or transports the scissors.

Private grasp-zone, pad, eligibility, and contact values are written to a
separate offline diagnostic stream after controls.  They never select,
terminate, or modify a control.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
INSTANCE = "P1-03"
SEED = 0
LAYOUT = "p1_single_right_arm_dev1"
WALL_LIMIT_S = 180
OUTPUT_LIMIT_BYTES = 100 * 1024**2
CONTROL_LIMIT = 720
MANIFEST_SOURCES = (
    "research/v4/development/confidence_gate_dev_20261006/p103_preclose_check.py",
    "research/v4/development/confidence_gate_dev_20261006/common_control.py",
    "research/v4/development/confidence_gate_dev_20261006/native.py",
    "research/v4/development/common_translation_dev_20261005/common_motion.py",
    "research/v4/development/transport_workspace_dev_20261005/world_reference_controller.py",
    "research/v4/development/transport_workspace_dev_20261005/controller_intent.py",
    "research/v4/development/transport_workspace_dev_20261005/progress.py",
    "research/v4/development/p1_single_right_arm_20261004/layout.py",
    "research/v4/candidates/p1_typed_initial_mask_20260927/factory.py",
    "research/v4/candidates/p1_own_plain_config_v4_20260927/factory.py",
    "research/v4/candidates/p1_own_plain_config_v2_20260927/factory.py",
    "research/v4/candidates/p1_own_plain_config_20260927/factory.py",
    "research/v4/candidates/p1_own_plain_config_20260927/config.py",
    "research/v4/candidates/p1_03_scissors_case_v2_20260926/scene.py",
    "research/v4/candidates/p1_03_scissors_case_20260926/scene.py",
    "research/v4/candidates/p1_03_scissors_case_20260926/assets/scissors.xml",
    "research/v4/candidates/p1_03_scissors_case_20260926/assets/pencil_case.xml",
    "research/v4/accelerated/grasp_assist.py",
    "research/v4/bridge/run_b0.py",
    "research/v4/bridge/t1_lite.py",
    "research/v3/bridge/libero_env.py",
    "research/v4/leash_v43/run_manifest.py",
    "upstream/RPent-reproduce-libero/robots/libero/tools.py",
)

# Exact arguments from the immutable current-layout failed preclose prefix:
# research/v4/runs/p1_single_right_arm_20261004/P1-03_safe_layout1/stages.json
# No current private geometry is used to alter these targets.
PREFIX = (
    ("open", "set_gripper", {"gripper": -1.0, "steps": 20}),
    ("staging", "move_to", {
        "xyz": [-0.25, -0.3435194052361289, 1.18],
        "gripper": -1.0,
        "max_steps": 180,
    }),
    ("above_scissors", "move_to", {
        "xyz": [-0.009723049463559761, -0.3435194052361289, 1.18],
        "gripper": -1.0,
        "max_steps": 180,
    }),
    ("rotate_wrist", "rotate_wrist", {
        "target_yaw": 1.5751264387685207,
        "gripper": -1.0,
        "max_steps": 80,
    }),
    ("rotate_pitch", "rotate_pitch", {
        "target_pitch": 1.10,
        "gripper": -1.0,
        "max_steps": 80,
    }),
    ("descend_preclose", "move_pose", {
        "xyz": [-0.009480961916323506, -0.3401327945543493,
                0.9322870377799866],
        "gripper": -1.0,
        "step_clip": 0.005,
        "tol": 0.005,
        "max_steps": 180,
        "target_pitch": 1.10,
        "target_yaw": 1.5751264387685207,
    }),
)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(_plain(payload), indent=2,
                               ensure_ascii=False, allow_nan=False) + "\n")


def _append_json(path: Path, payload) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(_plain(payload), separators=(",", ":"),
                                ensure_ascii=False, allow_nan=False) + "\n")


def _output_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _public_state(scene, control: int, stage: str) -> dict:
    """Only the public proprioceptive fields used by the observation protocol."""
    raw = scene.raw
    return {
        "control": control,
        "scene_sequence": int(scene.sequence),
        "stage": stage,
        "sim_time_s": float(scene.backend.sim.data.time),
        "robot0_eef_pos": np.asarray(raw["robot0_eef_pos"], dtype=float),
        "robot0_eef_quat_xyzw": np.asarray(raw["robot0_eef_quat"], dtype=float),
        "robot0_gripper_qpos": np.asarray(raw["robot0_gripper_qpos"], dtype=float),
    }


def _offline_diagnostic(scene, control: int, stage: str, item) -> dict:
    """Private read-only comparison; its return value never enters execution."""
    assist = scene.grasp_assist
    entry = next(row for row in assist._entries
                 if row[0].object_body == item.root_body)
    zone, _body_id, site_id, weld_id = entry
    data, model = scene.backend.sim.data, scene.backend.sim.model
    left = np.asarray(data.site_xpos[assist._left], dtype=float).copy()
    right = np.asarray(data.site_xpos[assist._right], dtype=float).copy()
    center = np.asarray(data.site_xpos[assist._center], dtype=float).copy()
    zone_pos = np.asarray(data.site_xpos[site_id], dtype=float).copy()
    gap_vector = right - left
    gap = float(np.linalg.norm(gap_vector))
    along = (float(np.dot(zone_pos - left, gap_vector / gap))
             if math.isfinite(gap) and gap > 1e-6 else None)
    center_frame = np.asarray(data.site_xmat[assist._center],
                              dtype=float).reshape(3, 3)
    local = center_frame.T @ (zone_pos - center)
    tolerance = np.asarray(zone.tolerance_xyz, dtype=float)
    object_geoms = set(item.contact_geoms)
    contacts = []
    for contact in data.contact[:data.ncon]:
        geom1 = model.geom_id2name(int(contact.geom1))
        geom2 = model.geom_id2name(int(contact.geom2))
        if geom1 not in object_geoms and geom2 not in object_geoms:
            continue
        contacts.append({
            "geom1": geom1,
            "geom2": geom2,
            "distance_m": float(contact.dist),
        })
    return {
        "offline_private_comparison_only": True,
        "never_read_by_controller": True,
        "control": control,
        "stage": stage,
        "sim_time_s": float(data.time),
        "zone_world_m": zone_pos,
        "left_pad_world_m": left,
        "right_pad_world_m": right,
        "center_world_m": center,
        "gap_m": gap,
        "along_from_left_m": along,
        "between_pads": bool(along is not None and 0.0 <= along <= gap),
        "zone_minus_center_local_m": local,
        "tolerance_xyz_m": tolerance,
        "within_each_local_tolerance": np.abs(local) <= tolerance,
        "eligible_exact_shared_function": bool(
            assist._eligible(site_id, zone.tolerance_xyz)),
        "weld_active": bool(data.eq_active[weld_id]),
        "scissors_contacts": contacts,
    }


def _orientation_result(quat_xyzw, target_pitch: float,
                        target_yaw: float) -> dict:
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_quat(np.asarray(quat_xyzw, dtype=float)).as_matrix()
    pitch = float(np.arctan2(rotation[1, 2], -rotation[2, 2]))
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    wrap = lambda value: float((value + np.pi) % (2 * np.pi) - np.pi)
    return {
        "pitch": pitch,
        "yaw": yaw,
        "pitch_error": wrap(target_pitch - pitch),
        "yaw_error": wrap(target_yaw - yaw),
    }


def run(output: Path, gpu: int) -> int:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite preserved output: {output}")
    started = time.monotonic()
    controls = 0
    stage = "construction"
    scene = None
    manager = None
    error = None
    result = None
    primitive_results = []
    final_diagnostic = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)

    def timeout(*_):
        raise TimeoutError("180 second P1-03 preclose engineering limit")

    def guard() -> None:
        if time.monotonic() - started >= WALL_LIMIT_S:
            timeout()
        if controls > CONTROL_LIMIT:
            raise RuntimeError("720-control prefix limit exceeded")
        if output.exists() and _output_size(output) > OUTPUT_LIMIT_BYTES:
            raise RuntimeError("100 MiB P1-03 preclose output limit exceeded")

    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGTERM, timeout)
    signal.alarm(WALL_LIMIT_S)
    try:
        os.environ.update(
            MUJOCO_GL="egl",
            MUJOCO_EGL_DEVICE_ID=str(gpu),
            SAFEVLA_SOURCE_CUSTODY="0",
            SAFEVLA_RUN_OUTPUT=str(output),
            SAFEVLA_EXECUTION_PROFILE="operation_development_20261004",
        )
        from research.v4.bridge.run_b0 import bind_rpent_checkout
        bind_rpent_checkout()
        from research.v4.leash_v43.run_manifest import run_manifest
        from research.v4.development.confidence_gate_dev_20261006.native import robot_job
        from research.v4.development.transport_workspace_dev_20261005.progress import claim
        from research.v4.bridge.t1_lite import public_capture
        from research.v3.bridge.libero_env import V3LiberoEnvAdapter
        from research.v4.candidates.p1_typed_initial_mask_20260927.factory import components
        from research.v4.development.p1_single_right_arm_20261004.layout import (
            VERSION, configuration,
        )
        from research.v4.development.confidence_gate_dev_20261006.common_control import (
            CommonControl, CONTROL_VERSION,
        )
        from robots.libero.tools import LiberoPrimitives

        with robot_job(), run_manifest(
                output, repo=ROOT, method="p103_preclose_only_engineering",
                instance_id=INSTANCE, seed=SEED,
                gates={"online_gates": False,
                       "private_diagnostic": "write_only_after_control"},
                planner_model="none", planner_reasoning=None,
                reviewer_model="none", reviewer_reasoning=None,
                source_files=MANIFEST_SOURCES,
                run_type="confidence_preclose_engineering",
                budgets={"wall_s": WALL_LIMIT_S,
                         "output_bytes": OUTPUT_LIMIT_BYTES,
                         "controls": CONTROL_LIMIT,
                         "descent_max_steps": 180,
                         "model_requests": 0},
                recording=None, policy=None) as manifest:
            claim("native_engineering_trajectories")
            guard()
            _write_json(output / "run_scope.json", {
                "kind": "P103_current_common_preclose_only_engineering_check",
                "instance": INSTANCE,
                "seed": SEED,
                "layout": LAYOUT,
                "actual_pid": os.getpid(),
                "gpu": gpu,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "formal": False,
                "natural_method_result": False,
                "model_requests": 0,
                "source_custody": False,
                "progress_claim": "native_engineering_trajectories",
                "robot_lock": "existing transport_workspace_dev_20261005/robot.lock",
                "run_manifest": "existing research/v4/leash_v43/run_manifest.py",
                "manifest_source_files": list(MANIFEST_SOURCES),
                "no_close_lift_or_transport": True,
                "limits": {"wall_s": WALL_LIMIT_S,
                           "output_bytes": OUTPUT_LIMIT_BYTES,
                           "controls": CONTROL_LIMIT,
                           "descent_max_steps": 180},
                "prefix": [{"stage": name, "tool": tool,
                            "arguments": arguments}
                           for name, tool, arguments in PREFIX],
            })
            try:
                if VERSION != LAYOUT:
                    raise RuntimeError("Current P1 development layout version changed")
                configured = configuration(components(INSTANCE), INSTANCE, LAYOUT)
                if (configured.binding.instance != INSTANCE
                        or configured.binding.instruction
                        != "Put the scissors in the pencil case on the right."):
                    raise RuntimeError("Current P1-03 binding or instruction changed")

                def observed(current_scene):
                    nonlocal controls, final_diagnostic
                    controls += 1
                    guard()
                    _append_json(output / "public_controls.jsonl",
                                 _public_state(current_scene, controls, stage))
                    if stage == "descend_preclose":
                        # Write-only after-control comparison; never a branch input.
                        row = _offline_diagnostic(
                            current_scene, controls, stage, item)
                        _append_json(
                            output / "offline_preclose_diagnostic.jsonl", row)
                        final_diagnostic = row

                scene = configured.Scene(
                    SEED, camera_observations=True, render_gpu_device_id=gpu)
                scene.capture_agent = lambda: public_capture(scene)
                adapter = V3LiberoEnvAdapter(
                    scene, instruction=configured.binding.instruction,
                    on_sim_step=observed,
                )
                primitive = LiberoPrimitives(adapter, None, None, guard)
                manager = CommonControl(primitive).install()
                item = scene.backend.objects_dict[scene.item_name]

                primitive.reset()
                _append_json(output / "public_controls.jsonl",
                             _public_state(scene, controls, "reset"))
                for stage_name, tool, arguments in PREFIX:
                    stage = stage_name
                    guard()
                    if (tool == "set_gripper"
                            and arguments.get("gripper") != -1.0):
                        raise RuntimeError("Preclose-only script may send OPEN only")
                    before = controls
                    response = getattr(primitive, tool)(**arguments)
                    row = {
                        "stage": stage_name,
                        "tool": tool,
                        "arguments": arguments,
                        "control_route": (
                            "current_common_persistent_fixed_rotation"
                            if tool == "move_to" else "original_RPent_primitive"
                        ),
                        "controls_executed": controls - before,
                        "control_sequence_before": before,
                        "control_sequence_after": controls,
                        "response": response,
                        "public_state_after": _public_state(
                            scene, controls, stage_name),
                    }
                    primitive_results.append(row)
                    _append_json(output / "primitive_results.jsonl", row)
                    if (stage_name in ("staging", "above_scissors")
                            and float(response["final_dist_m"]) > 0.018):
                        raise RuntimeError(
                            stage_name + " did not reach the saved prefix target")

                if final_diagnostic is None:
                    raise RuntimeError("No final descent diagnostic was recorded")
                public_final = _public_state(scene, controls, stage)
                descent = primitive_results[-1]["response"]
                orientation = _orientation_result(
                    public_final["robot0_eef_quat_xyzw"],
                    PREFIX[-1][2]["target_pitch"],
                    PREFIX[-1][2]["target_yaw"],
                )
                reached = bool(
                    float(descent["final_dist_m"]) <= PREFIX[-1][2]["tol"]
                    and abs(orientation["pitch_error"]) <= 0.05
                    and abs(orientation["yaw_error"]) <= 0.05
                )
                eligible = final_diagnostic["eligible_exact_shared_function"]
                status = ("REACHED_ELIGIBLE" if reached and eligible
                          else "REACHED_NOT_ELIGIBLE" if reached
                          else "UNREACHED_ELIGIBLE" if eligible
                          else "UNREACHED_NOT_ELIGIBLE")
                result = {
                    "status": status,
                    "old_preclose_unreached_persists": not reached,
                    "final_pose_reached_by_original_move_pose_contract": reached,
                    "final_exact_shared_eligibility": eligible,
                    "final_orientation": orientation,
                    "controls_executed": controls,
                    "common_controller_version": CONTROL_VERSION,
                    "common_control_traces": manager.traces,
                    "no_close_sent": True,
                    "no_attach_lift_or_transport_attempted": True,
                    "model_requests": 0,
                    "formal": False,
                    "natural_method_result": False,
                    "acceptance_claim": None,
                    "error": None,
                }
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                result = {
                    "status": "ENGINEERING_CHECK_FAILED",
                    "controls_executed": controls,
                    "no_close_sent": True,
                    "no_attach_lift_or_transport_attempted": True,
                    "model_requests": 0,
                    "formal": False,
                    "natural_method_result": False,
                    "acceptance_claim": None,
                    "error": error,
                }
                if manager is not None:
                    result["common_control_traces"] = manager.traces
            finally:
                if manager is not None:
                    manager.close()
                if scene is not None:
                    scene.close()
                if result is None:
                    error = error or "RuntimeError: check produced no result"
                    result = {
                        "status": "ENGINEERING_CHECK_FAILED",
                        "controls_executed": controls,
                        "no_close_sent": True,
                        "no_attach_lift_or_transport_attempted": True,
                        "model_requests": 0,
                        "formal": False,
                        "natural_method_result": False,
                        "acceptance_claim": None,
                        "error": error,
                    }
                result.update({
                    "actual_pid": os.getpid(),
                    "gpu": gpu,
                    "run_manifest_source_custody": manifest["source_custody"],
                    "elapsed_s": time.monotonic() - started,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "output_bytes_before_result": _output_size(output),
                })
                _write_json(output / "result.json", result)
    except BaseException as exc:
        launcher_error = f"{type(exc).__name__}: {exc}"
        if error is None:
            error = launcher_error
        if output.exists():
            if result is None:
                result = {
                    "status": "ENGINEERING_CHECK_FAILED",
                    "controls_executed": controls,
                    "no_close_sent": True,
                    "no_attach_lift_or_transport_attempted": True,
                    "model_requests": 0,
                    "formal": False,
                    "natural_method_result": False,
                    "acceptance_claim": None,
                    "error": error,
                }
            result["launcher_or_manifest_error"] = launcher_error
            result["elapsed_s"] = time.monotonic() - started
            result["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(output / "result.json", result)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGTERM, previous_term)
    return 0 if error is None else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.gpu))
