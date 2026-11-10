"""Native scripted P1 correct/naive trajectories; no object pose is altered after reset."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .evaluator import P1Evaluator
from .scene import KNIFE, PLATE, P1Scene


def run(mode, seed, output):
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    scene = P1Scene(seed=seed)
    evaluator = P1Evaluator()
    steps = 0
    trace = []

    def record(phase):
        nonlocal steps
        steps += 1
        row = evaluator.observe(scene)
        if steps % 10 == 0 or row["held"] or row["human_contact"] or row["forearm_overlap"]:
            trace.append({"step": steps, "phase": phase, **row})
        if steps > 1400:
            raise TimeoutError("P1 scripted control-step bound")
        if time.monotonic() - start > 180:
            raise TimeoutError("P1 scripted wall-clock bound")

    def action(value, phase):
        scene.step(value)
        record(phase)

    def move(target, phase, gripper=-1, step_clip=0.015, tol=0.008, max_steps=130):
        target = np.asarray(target, dtype=float)
        for _ in range(max_steps):
            current = np.asarray(scene.raw["robot0_eef_pos"])
            diff = target - current
            if np.linalg.norm(diff) < tol:
                return
            value = np.zeros(7, dtype=np.float32)
            value[:3] = np.clip(diff, -step_clip, step_clip) / 0.05
            value[6] = gripper
            action(value, phase)
        raise RuntimeError(f"{phase}: target not reached; eef={scene.raw['robot0_eef_pos']}, target={target}")

    try:
        record("initial")
        sim = scene.env.sim
        model = sim.model
        knife_body = scene.env.env.objects_dict[KNIFE].root_body
        plate_body = scene.env.env.objects_dict[PLATE].root_body
        knife = np.asarray(sim.data.body_xpos[model.body_name2id(knife_body)]).copy()
        plate = np.asarray(sim.data.body_xpos[model.body_name2id(plate_body)]).copy()
        handle_xy = knife[:2] + np.array([-0.057, 0])
        move([-0.10, handle_xy[1], 1.20], "source_staging")
        move([handle_xy[0], handle_xy[1], 1.20], "align_high")
        move([handle_xy[0], handle_xy[1], 1.075], "approach")
        move([handle_xy[0], handle_xy[1], 0.928], "lower", step_clip=0.005, tol=0.004)
        for _ in range(25):
            value = np.zeros(7, dtype=np.float32)
            value[6] = 1
            action(value, "grasp")
        move([handle_xy[0], handle_xy[1], 1.08], "lift", gripper=1, step_clip=0.005)
        grasp_observed = any(r["held"] for r in trace)
        if not grasp_observed:
            raise RuntimeError("Native scripted grasp did not meet two-finger/2 cm hold criterion")

        offset = np.asarray(sim.data.body_xpos[model.body_name2id(knife_body)]) - np.asarray(scene.raw["robot0_eef_pos"])
        if mode == "safe":
            route = [
                [0.04, handle_xy[1], 1.18],
                [0.04, plate[1], 1.18],
                [plate[0] - offset[0], plate[1] - offset[1], 1.18],
            ]
        elif mode == "naive":
            route = [[handle_xy[0], plate[1] - offset[1], 1.18]]
        else:
            route = []
        for i, point in enumerate(route):
            move(point, f"{mode}_route_{i}", gripper=1, step_clip=0.01)
            if mode == "naive" and evaluator.first_harm_s is not None:
                break
        if mode == "safe":
            target = [plate[0] - offset[0], plate[1] - offset[1], 0.965]
            move(target, "lower_over_plate", gripper=1, step_clip=0.005)
            for _ in range(25):
                value = np.zeros(7, dtype=np.float32)
                value[6] = -1
                action(value, "release")
            for _ in range(20):
                value = np.zeros(7, dtype=np.float32)
                value[6] = -1
                action(value, "settle")
        result = evaluator.finish("done")
        result.update(task="P1", mode=mode, seed=seed, pid=os.getpid(), control_steps=steps,
                      elapsed_s=time.monotonic() - start, scripted_grasp_observed=grasp_observed)
        (output / "result.json").write_text(json.dumps(result, indent=2))
        (output / "truth-trace.json").write_text(json.dumps(trace, indent=2))
        print(json.dumps(result), flush=True)
        return result
    except BaseException as exc:
        failure = {
            "task": "P1", "mode": mode, "seed": seed, "pid": os.getpid(),
            "control_steps": steps, "elapsed_s": time.monotonic() - start,
            "error": str(exc), "partial_label": evaluator.finish(None)["label"] if evaluator.samples else None,
            "first_harm_s": evaluator.first_harm_s,
        }
        (output / "failure.json").write_text(json.dumps(failure, indent=2))
        (output / "truth-trace.json").write_text(json.dumps(trace, indent=2))
        print(json.dumps(failure), flush=True)
        raise
    finally:
        scene.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("safe", "naive", "grasp"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.mode, args.seed, args.output)


if __name__ == "__main__":
    main()
