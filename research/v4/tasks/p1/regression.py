"""Provisional P1 correct/naive native scripts; never teleport objects after reset.

The inherited x=.35 source is known to challenge Panda reachability. A script
failure is preserved as a failed trial, not reinterpreted as task acceptance.
"""

import argparse
import json
import os
from pathlib import Path
import time

from PIL import Image
import numpy as np

from .evaluator import Evaluator
from .scene import Scene
from .template import freeze_inputs, load_instance, source_hashes


MAX_CONTROL_STEPS = 1400
MAX_WALL_S = 180.


def _yaw_from_quat(q):
    w, x, y, z = q
    return np.arctan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z))


def run(mode, instance_id, seed, output):
    if mode not in ("safe", "naive"):
        raise ValueError(mode)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    instance = load_instance(instance_id)
    hashes = source_hashes(instance, Path(__file__))
    freeze_inputs(output, hashes)
    started = time.monotonic()
    scene = None
    evaluator = Evaluator()
    controls = 0
    trace = []

    def observe(phase):
        nonlocal controls
        controls += 1
        state = evaluator.observe(scene)
        if controls % 10 == 0 or state["held"] or state["human_contact"] or state["forearm_footprint_overlap"]:
            trace.append({"control_step": controls, "phase": phase, **state})
        if controls > MAX_CONTROL_STEPS:
            raise TimeoutError("P1 script control-step cap")
        if time.monotonic() - started > MAX_WALL_S:
            raise TimeoutError("P1 script wall-time cap")
        return state

    def action(value, phase):
        scene.step(value)
        return observe(phase)

    def move(target, phase, gripper=-1, step_clip=.015, tol=.008, max_steps=130):
        target = np.asarray(target, dtype=float)
        for _ in range(max_steps):
            current = np.asarray(scene.raw["robot0_eef_pos"])
            diff = target - current
            if np.linalg.norm(diff) < tol:
                return
            value = np.zeros(7, dtype=np.float32)
            value[:3] = np.clip(diff, -step_clip, step_clip) / .05
            value[6] = gripper
            action(value, phase)
        raise RuntimeError(f"{phase}: EEF target not reached; current={scene.raw['robot0_eef_pos']}, target={target}")

    try:
        scene = Scene(instance_id=instance_id, seed=seed)
        observe("initial")
        packet = scene.capture_agent()
        for name, camera in packet["cameras"].items():
            Image.fromarray(camera["rgb"]).save(output / f"initial_{name}.png")
        Image.fromarray(scene.spectator_rgb()).save(output / "initial_spectator.png")
        sim = scene.env.sim
        item_obj = scene.env.env.objects_dict[instance.item["name"]]
        target_obj = scene.env.env.objects_dict[instance.target["name"]]
        item_qpos = np.asarray(sim.data.get_joint_qpos(item_obj.joints[0])).copy()
        yaw = _yaw_from_quat(item_qpos[3:7])
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        handle_xy = item_qpos[:2] + rotation @ instance.item["grasp_offset_xy_local_m"]
        target_qpos = np.asarray(sim.data.get_joint_qpos(target_obj.joints[0])).copy()
        move([-.10, handle_xy[1], 1.20], "source_staging")
        move([handle_xy[0], handle_xy[1], 1.20], "align_high")
        move([handle_xy[0], handle_xy[1], 1.075], "approach")
        move([handle_xy[0], handle_xy[1], instance.item["grasp_eef_z_m"]],
             "lower", step_clip=.005, tol=.004)
        for _ in range(25):
            value = np.zeros(7, dtype=np.float32)
            value[6] = 1
            action(value, "grasp")
        move([handle_xy[0], handle_xy[1], instance.item["lift_eef_z_m"]],
             "lift", gripper=1, step_clip=.005)
        grasp_observed = any(row["held"] for row in trace)
        if not grasp_observed:
            raise RuntimeError("P1 native grasp did not satisfy two-finger and >2 cm rule")
        item_body = item_obj.root_body
        offset = (np.asarray(sim.data.body_xpos[sim.model.body_name2id(item_body)])
                  - np.asarray(scene.raw["robot0_eef_pos"]))
        if mode == "safe":
            route = [[.04, handle_xy[1], 1.18],
                     [.04, target_qpos[1], 1.18],
                     [target_qpos[0] - offset[0], target_qpos[1] - offset[1], 1.18]]
        else:
            route = [[handle_xy[0], target_qpos[1] - offset[1], 1.18]]
        for i, waypoint in enumerate(route):
            move(waypoint, f"{mode}_transport_{i}", gripper=1, step_clip=.01)
            if mode == "naive" and evaluator.first_harm_s is not None:
                break
        if mode == "safe":
            move([target_qpos[0] - offset[0], target_qpos[1] - offset[1],
                  instance.target["place_eef_z_m"]],
                 "lower_over_target", gripper=1, step_clip=.005)
            for phase, count in (("release", 25), ("settle", 20)):
                for _ in range(count):
                    value = np.zeros(7, dtype=np.float32)
                    value[6] = -1
                    action(value, phase)
        result = evaluator.finish("done")
        result.update(instance=instance_id, mode=mode, seed=seed, pid=os.getpid(),
                      control_steps=controls, elapsed_s=time.monotonic() - started,
                      scripted_grasp_observed=grasp_observed, source_sha256=hashes)
        (output / "result.json").write_text(json.dumps(result, indent=2))
        (output / "truth-trace.json").write_text(json.dumps(trace, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}), flush=True)
        return result
    except BaseException as exc:
        failure = {"instance": instance_id, "mode": mode, "seed": seed, "pid": os.getpid(),
                   "control_steps": controls, "elapsed_s": time.monotonic() - started,
                   "error": str(exc), "partial_label": evaluator.finish(None)["label"] if evaluator.samples else None,
                   "first_harm_s": evaluator.first_harm_s, "source_sha256": hashes}
        (output / "failure.json").write_text(json.dumps(failure, indent=2))
        (output / "truth-trace.json").write_text(json.dumps(trace, indent=2))
        print(json.dumps({k: v for k, v in failure.items() if k != "source_sha256"}), flush=True)
        raise
    finally:
        if scene is not None:
            scene.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("safe", "naive"))
    parser.add_argument("--instance", default="P1-01")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.mode, args.instance, args.seed, args.output)


if __name__ == "__main__":
    main()
