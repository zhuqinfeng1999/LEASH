"""CPU-only geometric slot projection and Panda top/side IK qualification.

Run ``envs/libero/bin/python -m research.v4.accelerated.slot_preflight``.
This never renders, advances controls, calls a model, or labels a task result.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .tabletop import (IMAGE_HEIGHT, IMAGE_WIDTH, PILOTS, PUBLIC_VIEWS,
                       ROOT, SLOTS, create_scene, sample_slot)

# Conservative slot envelopes, not bounds of every future instance asset.
SLOT_HALF_XY = {
    "L": (.10, .04), "C": (.16, .13), "R": (.15, .15),
    "F": (.13, .10), "B": (.10, .10),
    "MICROWAVE": (.17, .15), "STOVE": (.16, .14),
}


def project(sim, camera: str, point, *, margin_px=10):
    m, d = sim.model, sim.data
    cid = m.camera_name2id(camera)
    local = np.asarray(d.cam_xmat[cid]).reshape(3, 3).T @ (np.asarray(point) - d.cam_xpos[cid])
    f = IMAGE_HEIGHT / (2 * math.tan(math.radians(float(m.cam_fovy[cid])) / 2))
    if local[2] >= -1e-6:
        return {"visible": False, "reason": "behind_camera"}
    u = IMAGE_WIDTH / 2 - f * local[0] / local[2]
    v = IMAGE_HEIGHT / 2 + f * local[1] / local[2]
    visible = (margin_px <= u < IMAGE_WIDTH - margin_px and
               margin_px <= v < IMAGE_HEIGHT - margin_px)
    return {"visible": bool(visible), "pixel": [round(float(u), 2), round(float(v), 2)],
            "depth_m": round(float(-local[2]), 4),
            "reason": "inside" if visible else "outside_frame"}


def project_at_joint_pose(sim, camera: str, point, joint_qpos, *, margin_px=10):
    """Project a point after a hypothetical IK pose, without moving live data."""
    model = sim.model._model
    data = mujoco.MjData(model)
    data.qpos[:] = sim.data.qpos
    data.qvel[:] = 0.
    for i, q in enumerate(joint_qpos, start=1):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"robot0_joint{i}")
        data.qpos[model.jnt_qposadr[jid]] = q
    mujoco.mj_forward(model, data)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    local = data.cam_xmat[cid].reshape(3, 3).T @ (np.asarray(point) - data.cam_xpos[cid])
    f = IMAGE_HEIGHT / (2 * math.tan(math.radians(float(model.cam_fovy[cid])) / 2))
    if local[2] >= -1e-6:
        return {"visible": False, "reason": "behind_camera"}
    u = IMAGE_WIDTH / 2 - f * local[0] / local[2]
    v = IMAGE_HEIGHT / 2 + f * local[1] / local[2]
    visible = (margin_px <= u < IMAGE_WIDTH - margin_px and
               margin_px <= v < IMAGE_HEIGHT - margin_px)
    return {"visible": bool(visible), "pixel": [round(float(u), 2), round(float(v), 2)],
            "depth_m": round(float(-local[2]), 4),
            "reason": "inside" if visible else "outside_frame"}


def projected_slot_envelope(sim, slot_name):
    """±2 cm, ±10° and 2–20 cm high box-corner projection, no occlusion."""
    slot = SLOTS[slot_name]
    hx, hy = SLOT_HALF_XY[slot_name]
    margins = {cam: float("inf") for cam in PUBLIC_VIEWS}
    visible = {cam: True for cam in PUBLIC_VIEWS}
    for dx in (-.02, .02):
        for dy in (-.02, .02):
            for yaw in (-math.pi / 18, math.pi / 18):
                co, si = math.cos(yaw), math.sin(yaw)
                for x in (-hx, hx):
                    for y in (-hy, hy):
                        for z in (.02, .20):
                            point = np.asarray(slot.xyz) + [dx + co * x - si * y,
                                                            dy + si * x + co * y, z]
                            for cam in PUBLIC_VIEWS:
                                sample = project(sim, cam, point)
                                visible[cam] &= sample["visible"]
                                if "pixel" in sample:
                                    u, v = sample["pixel"]
                                    margins[cam] = min(margins[cam], u, v,
                                                       IMAGE_WIDTH - u, IMAGE_HEIGHT - v)
    return {"half_xy_m": [hx, hy], "z_above_table_m": [.02, .20],
            "visible": visible,
            "minimum_border_margin_px": {k: round(v, 2) for k, v in margins.items()},
            "visible_both": all(visible.values())}


def _ik(sim, target, orientation, *, max_iter=100, position_tol=.012,
        rotation_tol=math.radians(12)):
    """Damped 6D Jacobian IK on a copied MjData; never changes the scene."""
    model = sim.model._model
    data = mujoco.MjData(model)
    data.qpos[:] = sim.data.qpos
    data.qvel[:] = 0.
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                            "gripper0_right_grip_site")
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"robot0_joint{i}") for i in range(1, 8)]
    qidx = np.asarray([model.jnt_qposadr[j] for j in joints])
    vidx = np.asarray([model.jnt_dofadr[j] for j in joints])
    limits = np.asarray(model.jnt_range[joints])
    q_home = data.qpos[qidx].copy()
    best = None
    for restart in (0, 1):
        data.qpos[qidx] = q_home if restart == 0 else np.mean(limits, axis=1)
        for iteration in range(max_iter):
            mujoco.mj_forward(model, data)
            p = np.asarray(data.site_xpos[site]).copy()
            R = np.asarray(data.site_xmat[site]).reshape(3, 3).copy()
            ep = np.asarray(target) - p
            er = Rotation.from_matrix(orientation @ R.T).as_rotvec()
            pos_error, rot_error = np.linalg.norm(ep), np.linalg.norm(er)
            if best is None or (pos_error + .06 * rot_error) < best[0]:
                best = (pos_error + .06 * rot_error, pos_error, rot_error, iteration)
            if pos_error <= position_tol and rot_error <= rotation_tol:
                return {"reachable": True, "position_error_m": round(float(pos_error), 5),
                        "rotation_error_deg": round(math.degrees(float(rot_error)), 2),
                        "iterations": iteration, "restart": restart,
                        "joint_qpos_rad": data.qpos[qidx].tolist()}
            jp = np.zeros((3, model.nv))
            jr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, site)
            J = np.vstack((jp[:, vidx], jr[:, vidx]))
            err = np.r_[ep, er]
            damped = J.T @ np.linalg.solve(J @ J.T + .025**2 * np.eye(6), err)
            data.qpos[qidx] = np.clip(data.qpos[qidx] + .55 * damped,
                                      limits[:, 0] + .005, limits[:, 1] - .005)
    return {"reachable": False, "position_error_m": round(float(best[1]), 5),
            "rotation_error_deg": round(math.degrees(float(best[2])), 2),
            "reason": "ik_no_convergence_or_joint_limit"}


def preflight_slots(task_id="R2-01", seed=0, *, include_devices=True):
    env = create_scene(task_id, seed, camera_observations=False)
    try:
        env.reset()
        sim = env.sim
        sid = sim.model.site_name2id("gripper0_right_grip_site")
        initial_R = np.asarray(sim.data.site_xmat[sid]).reshape(3, 3)
        # Top uses the neutral Panda orientation; side rolls 90° about X.
        # A Y pitch failed the central slot because of joint limits.
        orientations = {
            "top": initial_R,
            "side": Rotation.from_euler("x", -90, degrees=True).as_matrix() @ initial_R,
        }
        slot_names = ["L", "C", "R", "F", "B"]
        if include_devices:
            slot_names += ["MICROWAVE", "STOVE"]
        slots = {}
        for slot_name in slot_names:
            slot = SLOTS[slot_name]
            # Deterministic boundary certificate: nominal and all XY corners.
            samples = [(0., 0.)] + [(x, y) for x in (-.02, .02) for y in (-.02, .02)]
            projections, ik_results, active_wrist = [], {"top": [], "side": []}, []
            for dx, dy in samples:
                xyz = np.asarray(slot.xyz) + np.array([dx, dy, .10])
                projections.append({"offset_xy_m": [dx, dy],
                                    "views": {cam: project(sim, cam, xyz)
                                              for cam in PUBLIC_VIEWS}})
                if slot.graspable:
                    for approach, R in orientations.items():
                        result = _ik(sim, xyz, R)
                        ik_results[approach].append(result)
                        if approach == "top":
                            wrist = (project_at_joint_pose(
                                sim, "robot0_eye_in_hand",
                                np.asarray(slot.xyz) + [dx, dy, .05],
                                result["joint_qpos_rad"])
                                if result["reachable"] else
                                {"visible": False, "reason": "top_ik_failed"})
                            active_wrist.append(wrist)
            slots[slot_name] = {
                "point_test_z_m": round(slot.xyz[2] + .10, 3),
                "slot_envelope_projection": projected_slot_envelope(sim, slot_name),
                "projections": projections,
                "active_wrist_top_approach": active_wrist,
                "visible_agent_and_active_wrist_all_xy": (
                    all(p["views"]["agentview"]["visible"] for p in projections)
                    and all(p["visible"] for p in active_wrist)
                    if slot.graspable else None),
                "visible_both_all_xy": all(v["visible"] for p in projections
                                            for v in p["views"].values()),
                "ik": ik_results,
                "top_ik_all_xy": (all(r["reachable"] for r in ik_results["top"])
                                  if slot.graspable else None),
                "side_ik_all_xy": (all(r["reachable"] for r in ik_results["side"])
                                   if slot.graspable else None),
            }
        return {
            "task_id": task_id, "seed": seed, "tier": "CPU_GEOMETRY_ONLY",
            "camera_resolution": [IMAGE_WIDTH, IMAGE_HEIGHT],
            "public_views": list(PUBLIC_VIEWS),
            "slot_results": slots,
            "qualified_slots": [name for name, result in slots.items()
                                if result["visible_both_all_xy"] and
                                result["top_ik_all_xy"] is True and
                                result["side_ik_all_xy"] is True],
            "approach_qualified_slots": [name for name, result in slots.items()
                                         if result["visible_agent_and_active_wrist_all_xy"] is True
                                         and result["top_ik_all_xy"] is True
                                         and result["side_ik_all_xy"] is True],
            "limitations": [
                "Single 10-cm-above-slot point, not whole object silhouette or rendered pixels.",
                "Home-pose and top-approach wrist projections are separate; neither proves unoccluded full-object visibility.",
                "Jacobian IK only; path collision, grasp, controls and task outcome untested.",
                "Envelope projection samples ±10° and 2–20 cm height, but does not prove actual mesh silhouette, occlusion, or text pixels.",
            ],
        }
    finally:
        env.close()


if __name__ == "__main__":
    output = Path(__file__).with_name("slot_preflight_cpu.json")
    report = {task: preflight_slots(task) for task in PILOTS}
    output.write_text(json.dumps(report, indent=2))
    for task, result in report.items():
        print(task, "qualified:", result["qualified_slots"])
        print("failed:", [s for s in result["slot_results"] if s not in result["qualified_slots"]])
    print(output)
