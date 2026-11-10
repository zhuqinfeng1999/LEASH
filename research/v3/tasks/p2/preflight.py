"""Bounded P2 initial-frame capture; the three images are acceptance evidence."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .evaluator import P2Evaluator
from .scene import P2Scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene = P2Scene(seed=args.seed, size=args.size)
    try:
        evaluator = P2Evaluator()
        truth = evaluator.observe(scene)
        public = scene.observation()
        camera_stats = {}
        for name, item in public["cameras"].items():
            rgb = item["rgb"]
            depth = item["depth_m"]
            if rgb.shape != (args.size, args.size, 3) or depth.shape != (args.size, args.size):
                raise RuntimeError(f"{name}: wrong RGB-D dimensions")
            if not np.all(np.isfinite(depth)) or np.min(depth) <= 0:
                raise RuntimeError(f"{name}: invalid initial depth")
            Image.fromarray(rgb).save(args.output / f"{name}.png")
            camera_stats[name] = {
                "rgb_shape": list(rgb.shape),
                "depth_min_m": float(np.min(depth)),
                "depth_max_m": float(np.max(depth)),
            }
        Image.fromarray(scene.spectator_rgb()).save(args.output / "spectator.png")
        result = {
            "task": "P2", "seed": args.seed, "pid": os.getpid(),
            "initial_evaluator_only": truth,
            "initial_label": evaluator.finish(None)["label"],
            "public_camera_stats": camera_stats,
            "spectator_camera": "spectator",
        }
        (args.output / "preflight.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        scene.close()


if __name__ == "__main__":
    main()
