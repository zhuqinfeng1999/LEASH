"""Isolated P1 v6 candidate: 4 cm more knife-to-left-hand pickup clearance.

Inherits the accepted v5b natural person and common v4.2 assisted-grasp
physics. The only scene change is the initial knife y-position. This is a
candidate, not a promoted protocol or a method episode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.v4.candidates.p1_ma_fallback_20260925 import candidate_v5b as v5b


class P1V6Facade(v5b.P1V5BFacade):
    def _place_e1_layout_at_reset(self):
        super()._place_e1_layout_at_reset()
        sim = self.backend.sim
        item = self.backend.objects_dict[self.item_name]
        pose = np.asarray(sim.data.get_joint_qpos(item.joints[0]), dtype=float).copy()
        pose[1] += .040
        sim.data.set_joint_qpos(item.joints[0], pose)
        sim.data.set_joint_qvel(item.joints[0], np.zeros(6))
        sim.forward()


P1V6Evaluator = v5b.P1V5BEvaluator


def probe(mode: str, seed: int):
    with patch.object(v5b, "P1V5BFacade", P1V6Facade):
        row = v5b.probe(mode, seed)
    row.update(
        kind="p1_v6_four_cm_knife_clearance_cpu_candidate_not_method",
        scene_version="p1_v6_v5b_natural_person_knife_y_plus040",
    )
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("safe", "naive", "failure"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = probe(args.mode, args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"mode": args.mode, "seed": args.seed,
                      "label": result["result"]["label"] if result["result"] else None,
                      "accepted": result["accepted"], "error": result["error"],
                      "elapsed_s": round(result["elapsed_s"], 2)}, indent=2))
