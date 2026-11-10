"""Finite P3 zero-action native scene check and three initial camera images."""

import argparse
import json
import os
from pathlib import Path

from PIL import Image
import numpy as np

from .evaluator import P3Evaluator
from .scene import CUP, COASTER, P3Scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--cup-asset", choices=("original", "cup65"), default="original")
    parser.add_argument("--linger", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene = None
    report = {"task": "P3", "seed": args.seed, "pid": os.getpid(),
              "status": "failed", "section7_accepted": False, "cup_asset": args.cup_asset}
    try:
        scene = P3Scene(seed=args.seed, size=args.size, cup_asset=args.cup_asset)
        evaluator = P3Evaluator()
        truth = evaluator.observe(scene)
        packet = scene.observation()
        for name in scene.public_views:
            Image.fromarray(packet["cameras"][name]["rgb"]).save(args.output / f"{name}.png")
            np.save(args.output / f"{name}-depth-m.npy", packet["cameras"][name]["depth_m"])
        Image.fromarray(scene.spectator_rgb()).save(args.output / "spectator.png")
        sim = scene.env.sim
        objects = scene.env.env.objects_dict
        source = np.asarray(sim.data.get_joint_qpos(objects[CUP].joints[0]))[:3]
        goal = np.asarray(sim.data.get_joint_qpos(objects[COASTER].joints[0]))[:3]
        # Mathematical geometry check, independent of what the images show.
        delta = goal[:2] - source[:2]
        direct_intersects_laptop = abs(source[0]) < .155 and abs(goal[0]) < .155 and (
            min(source[1], goal[1]) < .11 and max(source[1], goal[1]) > -.11)
        report.update(status="zero_action_scene_rendered", initial_truth_evaluator_only=truth,
                      cup_start_xyz=source.tolist(), coaster_start_xyz=goal.tolist(),
                      direct_line_intersects_laptop=bool(direct_intersects_laptop),
                      agent_camera_shapes={name: list(packet["cameras"][name]["rgb"].shape)
                                           for name in scene.public_views})
        if args.linger:
            import time
            time.sleep(min(max(args.linger, 0), 10))
    except BaseException as exc:
        import traceback
        report.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        if scene is not None:
            scene.close()
        (args.output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "status": report["status"],
                          "error": report.get("error")}), flush=True)


if __name__ == "__main__":
    main()
