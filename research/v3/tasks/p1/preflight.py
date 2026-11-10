"""Bounded zero-action native scene check and three required initial images."""

import argparse
import json
import os
from pathlib import Path

from PIL import Image
import numpy as np

from .scene import P1Scene
from .evaluator import P1Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--linger", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scene = P1Scene(seed=args.seed, size=args.size)
    try:
        evaluator = P1Evaluator()
        truth = evaluator.observe(scene)
        packet = scene.observation()
        for name, camera in packet["cameras"].items():
            Image.fromarray(camera["rgb"]).save(args.output / (name + ".png"))
        Image.fromarray(scene.spectator_rgb()).save(args.output / "spectator.png")
        result = {
            "task": "P1", "seed": args.seed, "pid": os.getpid(),
            "initial_truth_evaluator_only": truth,
            "initial_eef_m": packet["robot_state"]["eef_pos"].tolist(),
            "agent_camera_shapes": {
                name: list(camera["rgb"].shape) for name, camera in packet["cameras"].items()
            },
            "spectator_camera": "spectator",
            "evaluator_initial_label": evaluator.finish(None)["label"],
        }
        (args.output / "preflight.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        if args.linger:
            import time
            time.sleep(min(args.linger, 15))
    finally:
        scene.close()


if __name__ == "__main__":
    main()
