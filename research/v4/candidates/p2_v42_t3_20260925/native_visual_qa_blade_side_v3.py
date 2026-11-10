"""Finite native RGB-D / natural-person QA for the isolated P2 candidate.

No planner, reviewer, GPT, SAM service or method episode is started. Spectator
is offline only. Simulator truth here is used solely for offline asset QA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from research.v4.accelerated.slot_preflight import project
from research.v4.candidates.p2_v42_t3_20260925.scene_blade_side_v3 import P2Facade
from research.v4.candidates.p2_v42_t3_20260925.evaluator import P2Evaluator
from research.v4.accelerated.tabletop import IMAGE_HEIGHT, IMAGE_WIDTH

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE_FILES = (
    HERE / "scene_blade_side_v3.py", HERE / "evaluator.py", HERE / "scripted_blade_side_v3.py",
    Path(__file__), ROOT / "research/v4/accelerated/tabletop.py",
    ROOT / "research/v4/accelerated/grasp_assist.py",
    ROOT / "research/configs/scenes/blunt_tool.xml",
    ROOT / "research/upstream/polyhaven/wood_table_diff_1k.png",
    HERE / "knife_brushed_dark_steel.xml", HERE / "knife_visible_blade.xml",
    HERE / "LICENSE.txt",
    ROOT / "research/safevla/person.py",
)


def hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in SOURCE_FILES}


def arm_angles(person):
    pose = person.pose(np.array([.08, -.35, .938]))
    out = {}
    for side in ("L", "R"):
        def at(name):
            key = f"{name}.{side}"
            rot, shift = pose[key]
            return rot @ person.head[key] + shift
        shoulder, elbow, wrist, knuckle = (at(name) for name in
                                           ("upperarm01", "lowerarm01", "wrist", "finger3-1"))
        def deg(a, b):
            return math.degrees(math.acos(float(np.clip(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))
        out[side] = {
            "elbow_deflection_deg": round(deg(elbow - shoulder, wrist - elbow), 1),
            "wrist_deflection_deg": round(deg(wrist - elbow, knuckle - wrist), 1),
            "shoulder_xyz_m": shoulder.tolist(),
            "elbow_xyz_m": elbow.tolist(),
            "wrist_xyz_m": wrist.tolist(),
            "knuckle_xyz_m": knuckle.tolist(),
        }
    return out


def run(seed, variant="dark_steel"):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("MUJOCO_EGL_DEVICE_ID") != "0":
        raise RuntimeError("P2 native visual QA requires explicit physical GPU0 mapping")
    out = HERE / f"native_qa_blade_side_v3_{variant}_seed{seed}"
    out.mkdir(exist_ok=False)
    before = hashes()
    started = time.monotonic()
    scene = P2Facade(seed, camera_observations=True, render_gpu_device_id=0)
    try:
        packet = scene.capture_agent()
        pid = os.getpid()
        pmon = subprocess.check_output(["nvidia-smi", "pmon", "-i", "0", "-c", "2"], text=True)
        own_pid_on_gpu0 = any(len(parts) > 1 and parts[0] == "0" and parts[1] == str(pid)
                              for parts in (line.split() for line in pmon.splitlines()))
        (out / "gpu0-pmon.txt").write_text(pmon)
        if not own_pid_on_gpu0:
            raise RuntimeError(f"new simulator PID {pid} not seen on physical GPU0")
        cameras = {}
        for name in scene.public_views:
            frame = packet["cameras"][name]
            rgb = np.asarray(frame["rgb"])
            depth = np.asarray(frame["depth_m"])
            if (rgb.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3) or
                    depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or
                    not np.isfinite(depth).all() or float(depth.min()) <= 0):
                raise RuntimeError(f"invalid native 640x480 RGB-D: {name}")
            Image.fromarray(rgb).save(out / f"{name}.png")
            np.savez_compressed(out / f"{name}-depth-calibration.npz",
                                depth_m=depth, intrinsics=frame["intrinsics"],
                                extrinsics=frame["extrinsics"])
            cameras[name] = {"rgb_shape": list(rgb.shape),
                             "depth_shape": list(depth.shape),
                             "depth_range_m": [float(depth.min()), float(depth.max())],
                             "frame_id": frame["frame_id"]}
        spectator = scene.spectator_rgb()
        Image.fromarray(spectator).save(out / "research_spectator.png")
        sim = scene.backend.sim
        model = sim.model
        item = scene.backend.objects_dict[scene.item_name]
        knife_id = model.body_name2id(item.root_body)
        palm_id = model.geom_name2id("proxy_forearm_1_person_palm_R")
        knife_xyz = np.asarray(sim.data.body_xpos[knife_id], dtype=float)
        palm_xyz = np.asarray(sim.data.geom_xpos[palm_id], dtype=float)
        projections = {name: {
            "knife": project(sim, name, knife_xyz + [0, 0, .03]),
            "right_palm": project(sim, name, palm_xyz),
        } for name in scene.public_views}
        protected = set(scene.person.protected)
        knife_geoms = set(item.contact_geoms)
        physical_human_contacts = []
        for contact in sim.data.contact[:sim.data.ncon]:
            if contact.dist > .001:
                continue
            a = model.geom_id2name(int(contact.geom1)) or ""
            b = model.geom_id2name(int(contact.geom2)) or ""
            if ((a in protected or b in protected) and
                    (a in knife_geoms or b in knife_geoms or
                     a.startswith(("robot0_", "gripper0_")) or
                     b.startswith(("robot0_", "gripper0_")))):
                physical_human_contacts.append({"a": a, "b": b,
                                                 "distance_m": float(contact.dist)})
        _, ids, all_vertices = scene.person.surface_vertices(np.array([.08, -.35, .938]))
        arm_skin = all_vertices[ids]
        over_table = (np.abs(arm_skin[:, 0]) <= .45) & (np.abs(arm_skin[:, 1]) <= .45)
        penetrating_table = over_table & (arm_skin[:, 2] < .898)
        initial_score = P2Evaluator().observe(scene)
        from research.v3.tasks.p2.evaluator import _world_aabb
        collision_gid = model.geom_name2id("blunt_tool_1_blade")
        visual_gid = model.geom_name2id("blunt_tool_1_blade_visual")
        collision_aabb = _world_aabb(sim, collision_gid)
        visual_aabb = _world_aabb(sim, visual_gid)
        blade_visual_check = {
            "collision_aabb_m": [x.tolist() for x in collision_aabb],
            "visual_aabb_m": [x.tolist() for x in visual_aabb],
            "aabb_match_at_1e_minus_8_m": all(np.allclose(x, y, atol=1e-8)
                                              for x, y in zip(collision_aabb, visual_aabb)),
            "collision_group": int(model.geom_group[collision_gid]),
            "visual_group": int(model.geom_group[visual_gid]),
            "visual_contype": int(model.geom_contype[visual_gid]),
            "visual_conaffinity": int(model.geom_conaffinity[visual_gid]),
            "visual_in_contact_geoms": "blunt_tool_1_blade_visual" in item.contact_geoms,
        }
        if not (blade_visual_check["aabb_match_at_1e_minus_8_m"] and
                blade_visual_check["collision_group"] == 0 and
                blade_visual_check["visual_group"] == 1 and
                blade_visual_check["visual_contype"] == 0 and
                blade_visual_check["visual_conaffinity"] == 0 and
                not blade_visual_check["visual_in_contact_geoms"]):
            raise RuntimeError("P2 visual blade changed collision or physical AABB")
        report = {
            "kind": "p2_native_visual_qa_not_method_episode",
            "seed": seed,
            "simulator_pid": pid,
            "physical_gpu0_sim_observed": own_pid_on_gpu0,
            "wall_seconds": round(time.monotonic() - started, 3),
            "source_sha256": before,
            "cameras": cameras,
            "spectator_shape": list(spectator.shape),
            "projections": projections,
            "human_pose": arm_angles(scene.person),
            "arm_skin_over_table_vertices": int(over_table.sum()),
            "arm_skin_below_table_minus2mm": int(penetrating_table.sum()),
            "arm_skin_min_z_over_table_m": float(arm_skin[over_table, 2].min()),
            "initial_human_contact_pairs": physical_human_contacts,
            "initial_harm": bool(initial_score["harm"]),
            "initial_tip_hand_distance_m": initial_score["tip_hand_surface_m"],
            "blade_visual_check": blade_visual_check,
            "knife_xyz_m": knife_xyz.tolist(),
            "right_palm_xyz_m": palm_xyz.tolist(),
            "note": "Visual inspection of all three PNGs is required before candidate promotion.",
        }
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"seed": seed, "pid": pid, "physical_gpu0": own_pid_on_gpu0,
                          "projections": projections,
                          "arm_skin_below_table_minus2mm": int(penetrating_table.sum()),
                          "initial_human_contacts": len(physical_human_contacts),
                          "initial_harm": bool(initial_score["harm"]),
                          "source_sha256_scene": before["research/v4/candidates/p2_v42_t3_20260925/scene_blade_side_v3.py"]}), flush=True)
    finally:
        scene.close()
    after = hashes()
    if after != before:
        raise RuntimeError("P2 source changed during native visual QA")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--variant", choices=("dark_steel", "wood_table", "visible_blade", "safe_slot_v2"), default="wood_table")
    args = parser.parse_args()
    run(args.seed, args.variant)
